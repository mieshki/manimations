"""  
LinearLayout Visualization Tool  
  
Hierarchical structure: CTA → Warp → Thread → Register  
Based on PyTorch tensors for real-world usage.  
"""  
  
import torch  
from triton._C.libtriton.gluon_ir import GluonOpBuilder  
from triton._C.libtriton import ir  
from dataclasses import dataclass  
from typing import List, Tuple, Dict  
  
  
# ============================================================================  
# CONFIGURATION  
# ============================================================================  
  
@dataclass  
class DeviceConfig:  
    """GPU device configuration."""  
    threads_per_warp: int = 32  # 32 for NVIDIA, 64 for AMD  
      
      
@dataclass  
class BlockedLayoutConfig:  
    """Layout configuration parameters."""  
    size_per_thread: List[int]  
    threads_per_warp: List[int]  
    warps_per_cta: List[int]  
    order: List[int]  
    ctas_per_cga: List[int] = None  
    cta_split_num: List[int] = None  
    cta_order: List[int] = None  
      
    def __post_init__(self):  
        rank = len(self.size_per_thread)  
        if self.ctas_per_cga is None:  
            self.ctas_per_cga = [1] * rank  
        if self.cta_split_num is None:  
            self.cta_split_num = [1] * rank  
        if self.cta_order is None:  
            self.cta_order = list(range(rank))  

    def get_tensor_shape(self):
        x1, y1 = self.size_per_thread
        x2, y2 = self.threads_per_warp
        x3, y3 = self.warps_per_cta

        x = x1 * x2 * x3 
        y = y1 * y2 * y3
        return x, y

    def get_layout_code_str(self):
    # Helper do formatowania list bez spacji po przecinku
        def fmt(lst):
            return "[" + ", ".join(map(str, lst)) + "]"

        code_str = f"""\
gl.BlockedLayout(
    size_per_thread={fmt(self.size_per_thread)},
    threads_per_warp={fmt(self.threads_per_warp)},
    warps_per_cta={fmt(self.warps_per_cta)},
    order={fmt(self.order)},
)
"""
        return code_str
  
  
# ============================================================================  
# CORE CLASSES - Hierarchical Structure  
# ============================================================================  
  
class Register:  
    """Represents a single register holding a tensor element."""  
      
    def __init__(self, reg_id: int, tensor_index: Tuple[int, ...]):  
        self.reg_id = reg_id  
        self.tensor_index = tensor_index  
      
    def __repr__(self):  
        idx_str = ", ".join(f"{idx:3d}" for idx in self.tensor_index)  
        return f"Register[{self.reg_id}] → tensor[{idx_str}]"  
  
  
class Thread:  
    """Represents a single thread (lane) in a warp."""  
      
    def __init__(self, lane_id: int, warp_id: int, block_id: int,   
                 bases_dict: Dict, num_registers: int):  
        self.lane_id = lane_id  
        self.warp_id = warp_id  
        self.block_id = block_id  
        self._bases_dict = bases_dict  
        self._num_registers = num_registers  
        self._registers = None  
      
    def get_registers(self) -> List[Register]:  
        """Get all registers owned by this thread."""  
        if self._registers is None:  
            self._registers = []  
            for reg_id in range(self._num_registers):  
                tensor_idx = self._compute_tensor_index(reg_id)  
                self._registers.append(Register(reg_id, tensor_idx))  
        return self._registers  
      
    def _compute_tensor_index(self, reg_id: int) -> Tuple[int, ...]:  
        """Compute tensor index using XOR arithmetic (GF(2))."""  
        rank = len(self._bases_dict['reg'][0]) if self._bases_dict['reg'] else len(self._bases_dict['lane'][0])  
        tensor_idx = [0] * rank  
          
        for dim_name, dim_id in [('reg', reg_id), ('lane', self.lane_id),  
                                  ('warp', self.warp_id), ('block', self.block_id)]:  
            bases = self._bases_dict[dim_name]  
            for bit_pos, basis in enumerate(bases):  
                if (dim_id >> bit_pos) & 1:  
                    for dim in range(rank):  
                        tensor_idx[dim] ^= basis[dim]  
          
        return tuple(tensor_idx)  
      
    def __repr__(self):  
        return f"Thread(lane={self.lane_id}, warp={self.warp_id}, block={self.block_id})"  
  
  
class Warp:  
    """Represents a warp (group of threads)."""  
      
    def __init__(self, warp_id: int, block_id: int, num_lanes: int,  
                 bases_dict: Dict, num_registers: int):  
        self.warp_id = warp_id  
        self.block_id = block_id  
        self._num_lanes = num_lanes  
        self._bases_dict = bases_dict  
        self._num_registers = num_registers  
        self._threads = None  
      
    def get_threads(self) -> List[Thread]:  
        """Get all threads in this warp."""  
        if self._threads is None:  
            self._threads = [  
                Thread(lane_id, self.warp_id, self.block_id,   
                       self._bases_dict, self._num_registers)  
                for lane_id in range(self._num_lanes)  
            ]  
        return self._threads  
      
    def __repr__(self):  
        return f"Warp[{self.warp_id}] ({self._num_lanes} threads)"  
  
  
class CTA:  
    """Represents a CTA (Cooperative Thread Array / thread block)."""  
      
    def __init__(self, block_id: int, num_warps: int, num_lanes: int,  
                 bases_dict: Dict, num_registers: int):  
        self.block_id = block_id  
        self._num_warps = num_warps  
        self._num_lanes = num_lanes  
        self._bases_dict = bases_dict  
        self._num_registers = num_registers  
        self._warps = None  
      
    def get_warps(self) -> List[Warp]:  
        """Get all warps in this CTA."""  
        if self._warps is None:  
            self._warps = [  
                Warp(warp_id, self.block_id, self._num_lanes,  
                     self._bases_dict, self._num_registers)  
                for warp_id in range(self._num_warps)  
            ]  
        return self._warps  
      
    def __repr__(self):  
        return f"CTA[{self.block_id}] ({self._num_warps} warps)"  
  
  
class BlockedLayoutAnalyzer:  
    """Analyzes and visualizes tensor layouts."""  
      
    def __init__(self, tensor: torch.Tensor, layout_config: BlockedLayoutConfig,  
                 device_config: DeviceConfig):  
        self.tensor = tensor  
        self.layout_config = layout_config  
        self.device_config = device_config  
          
        # Create linear layout  
        self._linear_layout = self._create_linear_layout()  
          
        # Compute derived properties  
        self._num_warps = self._compute_num_warps()  
        self._num_registers = 2 ** len(self._linear_layout.reg_bases)  
        self._num_blocks = self._compute_num_blocks()  
          
        # Create bases dictionary  
        self._bases_dict = {  
            'reg': self._linear_layout.reg_bases,  
            'lane': self._linear_layout.lane_bases,  
            'warp': self._linear_layout.warp_bases,  
            'block': self._linear_layout.block_bases  
        }  
      
    def _create_linear_layout(self):  
        """Create LinearLayout from configuration."""  
        context = ir.context()  
        ir.load_dialects(context)  
        builder = GluonOpBuilder(context)  
          
        layout_attr = builder.get_blocked_layout(  
            self.layout_config.size_per_thread,  
            self.layout_config.threads_per_warp,  
            self.layout_config.warps_per_cta,  
            self.layout_config.order,  
            self.layout_config.ctas_per_cga,  
            self.layout_config.cta_split_num,  
            self.layout_config.cta_order  
        )  
          
        return builder.to_linear_layout(layout_attr, list(self.tensor.shape))  
      
    def _compute_num_warps(self) -> int:  
        """Compute total number of warps."""  
        result = 1  
        for w in self.layout_config.warps_per_cta:  
            result *= w  
        return result  
      
    def _compute_num_blocks(self) -> int:  
        """Compute total number of blocks."""  
        result = 1  
        for b in self.layout_config.ctas_per_cga:  
            result *= b  
        return result  
      
    def get_ctas(self) -> List[CTA]:  
        """Get all CTAs."""  
        return [  
            CTA(block_id, self._num_warps, self.device_config.threads_per_warp,  
                self._bases_dict, self._num_registers)  
            for block_id in range(self._num_blocks)  
        ]  

    def get_cta_tensor_indexes(self, cta):
        idx = []
        warps = cta.get_warps()
        for w in warps:
            for t in w.get_threads():
                for r in t.get_registers():
                    idx.append(r.tensor_index)

        return idx

    def get_warp_tensor_indexes(self, warp):
        idx = []
        for t in warp.get_threads():
            for r in t.get_registers():
                idx.append(r.tensor_index)

        return idx

    def get_thread_tensor_indexes(self, thread):
        idx = []
        for r in thread.get_registers():
            idx.append(r.tensor_index)

        return idx


      
    def get_bases_info(self) -> Dict:  
        """Get bases information."""  
        return {  
            'register': self._linear_layout.reg_bases,  
            'lane': self._linear_layout.lane_bases,  
            'warp': self._linear_layout.warp_bases,  
            'block': self._linear_layout.block_bases  
        }  
      
    def get_summary(self) -> Dict:  
        """Get summary statistics."""  
        total_threads = self.device_config.threads_per_warp * self._num_warps * self._num_blocks  
        total_elements_covered = total_threads * self._num_registers  
        tensor_size = self.tensor.numel()  
          
        return {  
            'tensor_shape': list(self.tensor.shape),  
            'tensor_size': tensor_size,  
            'num_blocks': self._num_blocks,  
            'num_warps': self._num_warps,  
            'threads_per_warp': self.device_config.threads_per_warp,  
            'total_threads': total_threads,  
            'registers_per_thread': self._num_registers,  
            'total_elements_covered': total_elements_covered,  
            'coverage_ratio': total_elements_covered / tensor_size if tensor_size > 0 else 0  
        }  
  
  
# ============================================================================  
# VISUALIZATION  
# ============================================================================  
  
class LayoutPrinter:  
    """Handles printing and visualization of layout information."""  
      
    def __init__(self, analyzer: BlockedLayoutAnalyzer):  
        self.analyzer = analyzer  
      
    def print_all(self, max_threads_per_warp: int = 8, max_regs_per_thread: int = 8):  
        """Print complete layout visualization."""  
        self._print_header()  
        self._print_tensor_info()  
        self._print_layout_config()  
        self._print_bases()  
        self._print_hierarchy_sample(max_threads_per_warp, max_regs_per_thread)  
        self._print_summary()  
      
    def _print_section(self, title: str, char: str = "=", width: int = 80):  
        """Print section header."""  
        print(f"\n{char * width}")  
        print(title)  
        print(f"{char * width}")  
      
    def _print_header(self):  
        """Print main header."""  
        self._print_section("TRITON LINEAR LAYOUT ANALYZER")  
      
    def _print_tensor_info(self):  
        """Print tensor information."""  
        self._print_section("TENSOR INFORMATION", "-")  
        print(f"Shape: {list(self.analyzer.tensor.shape)}")  
        print(f"Size: {self.analyzer.tensor.numel()} elements")  
        print(f"Dtype: {self.analyzer.tensor.dtype}")  
        print(f"Device: {self.analyzer.tensor.device}")  
      
    def _print_layout_config(self):  
        """Print layout configuration."""  
        self._print_section("LAYOUT CONFIGURATION", "-")  
        config = self.analyzer.layout_config  
        print(f"Size per thread: {config.size_per_thread}")  
        print(f"Threads per warp: {config.threads_per_warp}")  
        print(f"Warps per CTA: {config.warps_per_cta}")  
        print(f"Order: {config.order}")  
        print(f"CTAs per CGA: {config.ctas_per_cga}")  
        print(f"CTA split num: {config.cta_split_num}")  
        print(f"CTA order: {config.cta_order}")  
      
    def _print_bases(self):  
        """Print bases information."""  
        self._print_section("LINEAR LAYOUT BASES", "-")  
        bases_info = self.analyzer.get_bases_info()  
          
        for name, bases in [('Register', bases_info['register']),  
                           ('Lane', bases_info['lane']),  
                           ('Warp', bases_info['warp']),  
                           ('Block', bases_info['block'])]:  
            print(f"\n{name} bases: {len(bases)} bases")  
            if bases:  
                for i, basis in enumerate(bases):  
                    print(f"  basis[{i}]: {basis}")  
            else:  
                print("  (empty)")  
      
    def _print_hierarchy_sample(self, max_threads: int, max_regs: int):  
        """Print sample of hierarchical structure."""  
        self._print_section("HIERARCHICAL STRUCTURE SAMPLE", "-")  
          
        ctas = self.analyzer.get_ctas()  
          
        for cta in ctas[:1]:  # Show first CTA only  
            print(f"\n{cta}")  
            warps = cta.get_warps()  
              
            for warp in warps[:2]:  # Show first 2 warps  
                print(f"  {warp}")  
                threads = warp.get_threads()  
                  
                for thread in threads[:max_threads]:  # Show limited threads  
                    print(f"    {thread}")  
                    registers = thread.get_registers()  
                      
                    for register in registers[:max_regs]:  # Show limited registers  
                        print(f"      {register}")  
                      
                    if len(registers) > max_regs:  
                        print(f"      ... ({len(registers) - max_regs} more registers)")  
                  
                if len(threads) > max_threads:  
                    print(f"    ... ({len(threads) - max_threads} more threads)")  
      
    def _print_summary(self):  
        """Print summary statistics."""  
        self._print_section("SUMMARY", "-")  
        summary = self.analyzer.get_summary()  
          
        print(f"Tensor shape: {summary['tensor_shape']}")  
        print(f"Tensor size: {summary['tensor_size']} elements")  
        print(f"Number of CTAs: {summary['num_blocks']}")  
        print(f"Warps per CTA: {summary['num_warps']}")  
        print(f"Threads per warp: {summary['threads_per_warp']}")  
        print(f"Total threads: {summary['total_threads']}")  
        print(f"Registers per thread: {summary['registers_per_thread']}")  
        print(f"Total elements covered: {summary['total_elements_covered']}")  

if __name__ == "__main__":
    t1 = torch.arange(0, 64*16,1, device='cpu').reshape(64, 16)
    layout_config = BlockedLayoutConfig(
                        [2, 4],
                        [16, 2],
                        [2, 2],
                        [1, 0],
                        [1, 1],
                        [1, 1],
                        [0, 1]
                    )
    device_config = DeviceConfig() 
    layout_analyzer = BlockedLayoutAnalyzer(t1, layout_config, device_config)
    layout_printer = LayoutPrinter(layout_analyzer)

    layout_printer.print_all()
    
    ctas = layout_analyzer.get_ctas()
    warps = ctas[0].get_warps()
    threads = warps[0].get_threads()
    registers = threads[0].get_registers()

    print(f'CTAS={len(ctas)}')
    print(f'Warps={len(warps)}')
    print(f'Threads={len(threads)}')
    print(f'Registers={len(registers)}')


    cta = layout_analyzer.get_ctas()[0]
    warp = cta.get_warps()[0]
    thread = warp.get_threads()[0]

    registers = thread.get_registers()
    print(registers)
    print("Values for CTA[0] W[0] T[0]")
    for reg in registers:
        print(f'[{reg.reg_id}] - [{reg.tensor_index}] - val:{t1[reg.tensor_index]}')


from gluon_linear_mapping import BlockedLayoutAnalyzer, BlockedLayoutConfig, DeviceConfig  
from triton.experimental.gluon.language._layouts import bases_per_dim  
import numpy as np  
import torch  

from pudb import set_trace
  
  
def compute_tensor_index(block_id, warp_id, lane_id, reg_id, bases_dict, rank):  
    """  
    Oblicza tensor index dla danej pozycji w hierarchii GPU.  
      
    Args:  
        block_id: ID bloku (CTA)  
        warp_id: ID warpa  
        lane_id: ID wątku (lane) w warpie  
        reg_id: ID rejestru w wątku  
        bases_dict: Słownik z bazami dla każdego poziomu  
        rank: Liczba wymiarów tensora  
      
    Returns:  
        Lista z indeksem tensora [x, y, ...]  
    """  
    tensor_idx = [0] * rank  
      
    # Iteruj przez wszystkie poziomy hierarchii  
    for dim_name, dim_id in [('reg', reg_id), ('lane', lane_id),   
                              ('warp', warp_id), ('block', block_id)]:  
        bases = bases_dict[dim_name]  
          
        # Dla każdego bitu w dim_id, XOR odpowiednią bazę  
        for bit_pos, basis in enumerate(bases):  
            if (dim_id >> bit_pos) & 1:  # Sprawdź czy bit jest ustawiony  
                for dim in range(rank):  
                    tensor_idx[dim] ^= basis[dim]  
      
    return tensor_idx  
  
class LinearLayoutAnalyzer:
    def __init__(self, tensor, linear_layout):
        self.tensor = tensor
        self.linear_layout = linear_layout

        self.rank = len(linear_layout.shape)    
    
        # Oblicz metryki    
        self.registers_per_lane = bases_per_dim(linear_layout.reg_bases, self.rank)    
        self.lanes_per_warp = bases_per_dim(linear_layout.lane_bases, self.rank)    
        self.warps_per_cta = bases_per_dim(linear_layout.warp_bases, self.rank)    

    def get_reg_bases(self):
        return self.linear_layout.reg_bases

    def get_registers_count(self):
        return np.prod(self.registers_per_lane)  

    def get_lane_bases(self):
        return self.linear_layout.lane_bases

    def get_warp_bases(self):
        return self.linear_layout.warp_bases

    def get_block_bases(self):
        return self.linear_layout.block_bases

    def get_linear_layout_bases_str(self):
        # helper do formatowania list bez spacji po przecinku
        def fmt_list(lst):
            if not lst:
                return "[]"
            if isinstance(lst[0], (list, tuple)):
                return "[" + ", ".join(fmt_list(x) for x in lst) + "]"
            else:
                return "[" + ", ".join(map(str, lst)) + "]"

        ll = self.linear_layout

        code_str = f"""\
DistributedLinearLayout(
    reg_bases={fmt_list(ll.reg_bases)},
    lane_bases={fmt_list(ll.lane_bases)},
    warp_bases={fmt_list(ll.warp_bases)},
    block_bases={fmt_list(ll.block_bases)},
    shape={fmt_list(ll.shape)},
)
"""
        return code_str

    def print(self): 
        print(f"Registers per lane (size_per_thread): {self.registers_per_lane}")    
        print(f"Lanes per warp (threads_per_warp): {self.lanes_per_warp}")    
        print(f"Warps per CTA (warps_per_cta): {self.warps_per_cta}")    
        print(f"\nTotal warps: {np.prod(self.warps_per_cta)}")    
        print(f"Total threads per CTA: {np.prod(self.lanes_per_warp) * np.prod(self.warps_per_cta)}")  
          
        # Przygotuj słownik z bazami  
        bases_dict = {  
            'reg': self.linear_layout.reg_bases,  
            'lane': self.linear_layout.lane_bases,  
            'warp': self.linear_layout.warp_bases,  
            'block': self.linear_layout.block_bases  
        }  
          
        # Wyświetl bazy  
        print("\n" + "="*80)  
        print("LINEAR LAYOUT BASES")  
        print("="*80)  
        print(f"reg_bases:   {self.linear_layout.reg_bases}")  
        print(f"lane_bases:  {self.linear_layout.lane_bases}")  
        print(f"warp_bases:  {self.linear_layout.warp_bases}")  
        print(f"block_bases: {self.linear_layout.block_bases}")  
          
        # Oblicz tensor indices dla b=0, w=0, t=0 (lane=0)  
        block_id = 0  
        warp_id = 0  
        lane_id = 0  
          
        print("\n" + "="*80)  
        print(f"TENSOR INDICES dla Block={block_id}, Warp={warp_id}, Thread={lane_id}")  
        print("="*80)  
          
        num_registers = np.prod(self.registers_per_lane)  
        print(f"\nLiczba rejestrów na wątek: {num_registers}")  
        print(f"\nTensor indices [x, y] dla każdego rejestru:\n")  
          
        for reg_id in range(num_registers):  
            tensor_idx = compute_tensor_index(block_id, warp_id, lane_id, reg_id,   
                                             bases_dict, self.rank)  
            value = t1[tensor_idx[0], tensor_idx[1]].item()  
            print(f"  Register[{reg_id}] → tensor[{tensor_idx[0]:2d}, {tensor_idx[1]:2d}] = {value:4d}")  
          
        # Dodatkowa wizualizacja - pokaż jak XOR działa dla pierwszych kilku rejestrów  
        print("\n" + "="*80)  
        print("SZCZEGÓŁOWA WIZUALIZACJA XOR (pierwsze 3 rejestry)")  
        print("="*80)  
          
        for reg_id in range(min(3, num_registers)):  
            print(f"\n--- Register {reg_id} (binary: {bin(reg_id)}) ---")  
            tensor_idx = [0, 0]  
              
            print(f"Start: tensor_idx = {tensor_idx}")  
              
            # Register level  
            for bit_pos, basis in enumerate(bases_dict['reg']):  
                if (reg_id >> bit_pos) & 1:  
                    print(f"  reg bit {bit_pos} = 1 → XOR reg_bases[{bit_pos}] = {basis}")  
                    for dim in range(self.rank):  
                        tensor_idx[dim] ^= basis[dim]  
                    print(f"    tensor_idx = {tensor_idx}")  
              
            # Lane level (lane_id = 0, więc wszystkie bity = 0)  
            print(f"  lane_id = {lane_id} (0b{bin(lane_id)[2:].zfill(5)}) → brak XOR")  
              
            # Warp level (warp_id = 0, więc wszystkie bity = 0)  
            print(f"  warp_id = {warp_id} (0b{bin(warp_id)[2:].zfill(2)}) → brak XOR")  
              
            # Block level (block_id = 0, brak baz)  
            print(f"  block_id = {block_id} → brak baz")  
              
            print(f"Wynik końcowy: tensor[{tensor_idx[0]}, {tensor_idx[1]}] = {t1[tensor_idx[0], tensor_idx[1]].item()}")
  

if __name__ == "__main__":  
    layout_config = BlockedLayoutConfig(  
                        [2, 4],  
                        [16, 2],  
                        [2, 2],  
                        [1, 0],  
                        [1, 1],  
                        [1, 1],  
                        [0, 1]  
                    )  
    tensor_shape = layout_config.get_tensor_shape()
    rows, cols = tensor_shape
    t1 = torch.arange(rows * cols, dtype=torch.int).reshape(tensor_shape)

    device_config = DeviceConfig()   
    layout_analyzer = BlockedLayoutAnalyzer(t1, layout_config, device_config)  
    linear_layout = layout_analyzer._linear_layout  

    ll_analyzer = LinearLayoutAnalyzer(t1, linear_layout)
    ll_analyzer.print()
    set_trace()
    t = 2
  

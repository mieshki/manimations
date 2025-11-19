from manim import *
from manim import config
from manim import Scene, VGroup, Text, Rectangle, Code, BraceLabel
from manim import BLUE, ORANGE, YELLOW, BLUE_D
from manim import LEFT, RIGHT, UP, DOWN
from manim import Create, Uncreate, FadeIn, FadeOut, Write, GrowFromCenter, SurroundingRectangle, smooth

import torch
from gluon_linear_mapping import BlockedLayoutConfig, DeviceConfig, BlockedLayoutAnalyzer
from pudb import set_trace


class BlockedLayout2D(Scene):
    PADDING = 0.5
    TOTAL_PANES = 3
    FRAME_HEIGHT_FACTOR = 0.98

    def place_in_pane(self, obj, pane_index, padding=0.1, height_bottom_padding = None, vert_align="MID", 
                      hor_align="MID", animate=False, scale=1):
        """Place object in vertical pane with alignment options."""
        frame_width = config.frame_width
        frame_height = config.frame_height
        pane_width = frame_width / self.TOTAL_PANES

        # Calculate horizontal position
        left_edge = -frame_width / 2
        center_x = left_edge + (pane_index + 0.5) * pane_width
        
        x_offsets = {
            "LEFT": -pane_width/2 + padding + (obj.width * scale)/2,
            "RIGHT": pane_width/2 - padding - (obj.width * scale)/2,
            "MID": 0
        }
        x = center_x + x_offsets[hor_align]

        # Calculate vertical position
        y_offsets = {
            "UP": frame_height/2 - padding - (obj.height * scale)/2,
            "DOWN": -frame_height/2 + (padding if height_bottom_padding is None else height_bottom_padding) + (obj.height * scale)/2,
            "MID": 0
        }
        y = y_offsets[vert_align]

        if animate:
            animation = obj.animate.scale(scale).move_to([x, y, 0]) if scale != 1 else obj.animate.move_to([x, y, 0])
            self.play(animation)
        else:
            obj.move_to([x, y, 0])

    def _create_indices(self, tensor, is_row, count):
        """Create row or column indices."""
        indices = VGroup()
        for i in range(count):
            target = tensor.elements[i][0] if is_row else tensor.elements[0][i]
            idx = Text(str(i), font_size=8)
            direction = LEFT if is_row else UP
            idx.next_to(target, direction, buff=0.05)
            indices.add(idx)
        return indices

    def print_tensor_row_cols(self, tensor):
        """Add row and column indices to tensor."""
        rows, cols = tensor.shape
        row_indices = self._create_indices(tensor, True, rows)
        col_indices = self._create_indices(tensor, False, cols)
        
        tensor.row_indices_text = row_indices
        tensor.col_indices_text = col_indices
        self.play(Write(col_indices), Write(row_indices))

    def add_title(self, msg, item, font_size=22, color=BLUE):
        """Add title above item."""
        title = Text(msg, font_size=font_size, color=color)
        self.place_in_pane(title, 0, vert_align="UP")
        title.next_to(item, UP)
        self.play(Write(title))

    def _calculate_cell_dimensions(self, rows, cols):
        """Calculate cell dimensions for tensor grid."""
        frame_width = config.frame_width / self.TOTAL_PANES
        # Reserve space for bottom brace label
        self.brace_height = 0.75
        frame_height = config.frame_height * self.FRAME_HEIGHT_FACTOR - self.brace_height
        cell_width = (frame_width - self.PADDING * 2) / cols
        cell_height = (frame_height - self.PADDING * 2) / rows
        return cell_width, cell_height

    def _create_cell(self, value, cell_width, cell_height, print_values):
        """Create single tensor cell."""
        square = Rectangle(height=cell_height, width=cell_width, 
                          stroke_width=1, color=BLUE)
        if print_values:
            text = Text(f"{value}", font_size=5)
            text.move_to(square.get_center())
            return VGroup(square, text)
        return VGroup(square)

    def _attach_metadata(self, vgroup, obj):
        """Attach object metadata to VGroup."""
        for k, v in obj.__dict__.items():
            setattr(vgroup, k, v)
        vgroup.model = obj
        return vgroup

    def _create_hierarchy(self, tensor_elements):
        """Create CTA -> Warp -> Thread -> Register hierarchy."""
        ctas_group = VGroup()
        
        for cta in self.layout_analyzer.get_ctas():
            warps_group = VGroup()
            
            for warp in cta.get_warps():
                threads_group = VGroup()
                
                for thread in warp.get_threads():
                    register_group = VGroup()
                    
                    for reg in thread.get_registers():
                        x, y = reg.tensor_index
                        reg_elem = tensor_elements[x][y]
                        reg_group = self._attach_metadata(VGroup(reg_elem), reg)
                        register_group.add(reg_group)
                    
                    thread.registers = register_group
                    thread_group = self._attach_metadata(VGroup(*register_group), thread)
                    threads_group.add(thread_group)
                
                warp_group = self._attach_metadata(VGroup(*threads_group), warp)
                warp_group.threads = threads_group
                warps_group.add(warp_group)
            
            cta_group = self._attach_metadata(VGroup(*warps_group), cta)
            cta_group.warps = warps_group
            ctas_group.add(cta_group)
        
        return ctas_group

    def _add_brace_labels(self, obj, rows_text, cols_text, rows_direction=DOWN, cols_direction=RIGHT,
                          rows_shift=(0, 0.25), cols_shift=(0.20, 0)):
        """Add brace labels to object with configurable directions and shifts."""
        # Cols brace (vertical - typically on side)
        obj.brace_label_cols = BraceLabel(obj, text=str(rows_text), brace_direction=cols_direction, buff=0.05)
        obj.brace_label_cols.label.shift(LEFT * cols_shift[0] + UP * cols_shift[1])
        
        # Rows brace (horizontal - typically on bottom)
        obj.brace_label_rows = BraceLabel(obj, text=str(cols_text), brace_direction=rows_direction, buff=0.05)
        obj.brace_label_rows.label.shift(LEFT * rows_shift[0] + UP * rows_shift[1])

    def _show_brace_labels(self, obj, run_time=1.5):
        """Show brace labels with animation."""
        self.play(
            GrowFromCenter(obj.brace_label_cols),
            GrowFromCenter(obj.brace_label_rows),
            run_time=run_time
        )

    def _hide_brace_labels(self, obj, run_time=1.0):
        """Hide brace labels with animation."""
        self.play(
            FadeOut(obj.brace_label_cols),
            FadeOut(obj.brace_label_rows),
            run_time=run_time
        )

    def create_tensor(self, print_values=False, print_tensor_dims=True):
        """Create tensor visualization with hierarchical structure."""
        rows, cols = self.torch_tensor.shape
        cell_width, cell_height = self._calculate_cell_dimensions(rows, cols)
        
        # Create tensor grid
        tensor_elements = [[None for _ in range(cols)] for _ in range(rows)]
        tensor = VGroup()
        
        for i in range(rows):
            for j in range(cols):
                value = self.torch_tensor[i, j].item()
                cell = self._create_cell(value, cell_width, cell_height, print_values)
                tensor_elements[i][j] = cell[0] if print_values else cell[0][0]
                tensor.add(cell)

        # Create hierarchical structure
        tensor.ctas = self._create_hierarchy(tensor_elements)
        
        # Arrange and configure tensor
        tensor.arrange_in_grid(rows=rows, cols=cols, buff=0)
        tensor.cell_size = max(cell_width, cell_height)
        tensor.shape = self.torch_tensor.shape
        tensor.elements = tensor_elements

        # Place and display
        self.place_in_pane(tensor, 0, padding=self.PADDING, height_bottom_padding=self.brace_height, vert_align="DOWN")
        self.play(FadeIn(tensor))

        if print_tensor_dims:
            self.print_tensor_row_cols(tensor)

        # Add and show brace labels
        self._add_brace_labels(tensor, rows, cols)
        self._show_brace_labels(tensor)
        
        return tensor

    def init_layout_analyzer(self):
        """Initialize layout analyzer with configuration."""
        self.layout_config = BlockedLayoutConfig(
            [2, 4], [16, 2], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]
        )
        tensor_shape = self.layout_config.get_tensor_shape()
        rows, cols = tensor_shape
        self.torch_tensor = torch.arange(rows * cols, dtype=torch.int).reshape(tensor_shape)
        
        device_config = DeviceConfig()
        self.layout_analyzer = BlockedLayoutAnalyzer(
            self.torch_tensor, self.layout_config, device_config
        )

    def highlight_code_line(self, lineno):
        self.unhighlight_code_line()
        line = self.code[1][lineno]
        surrounding_rect = SurroundingRectangle(
            line,
            color=BLUE_D,
            fill_color=BLUE_D,
            fill_opacity=0.05,
            buff=0.01,
            stroke_width=1
        )

        self.play(Create(surrounding_rect), run_time=1)
        self.code.current_highlight = surrounding_rect

    def unhighlight_code_line(self):
        if hasattr(self.code, 'current_highlight'):
            self.play(FadeOut(self.code.current_highlight))

    def create_code_snippet(self):
        """Create and display code snippet in right panel."""
        code_str = self.layout_config.get_layout_code_str()
        code = Code(
            code_string=code_str,
            language="python",
            background="rectangle",
            add_line_numbers=False,
            background_config={"stroke_color": "maroon"},
        )
        
        code_target_width = config.frame_width / self.TOTAL_PANES - self.PADDING
        code.scale(code_target_width / code.width)
        self.place_in_pane(code, 2, padding=self.PADDING, vert_align="UP")
        self.play(Write(code))
        self.code = code

    def init(self):
        """Initialize scene components."""
        self.init_layout_analyzer()
        self.create_code_snippet()
        self.tensor = self.create_tensor()
        self.add_title("Warps per CTA", self.tensor)

    def _create_labels_and_rects(self, items, label_prefix, font_size, id_attr):
        """Create labels and surrounding rectangles for items."""
        anims = []
        rects = []
        labels = []
        
        for item in items:
            item_id = getattr(item, id_attr)
            label = Text(f"{label_prefix} {item_id}", font_size=font_size)
            label.move_to(item.get_center())
            labels.append(label)
            anims.append(Write(label))
            
            rect = SurroundingRectangle(item, color=YELLOW, buff=0, stroke_width=1)
            rects.append(rect)
            anims.append(Create(rect))
        
        return anims, rects, labels

    def _fade_out_all(self, *groups):
        """Fade out all items in groups."""
        anims = [FadeOut(item) for group in groups for item in group]
        self.play(*anims)

    def _highlight_and_focus(self, obj, color=ORANGE, scale_factor=1.2):
        """Highlight object with color and focus animation."""
        self.play(obj.animate.set_color(color))
        self.play(obj.animate.scale(scale_factor), run_time=0.3, rate_func=smooth)
        self.play(obj.animate.scale(1/scale_factor), run_time=0.3, rate_func=smooth)

    def _scale_to_pane(self, obj, pane_index, vert_align="DOWN"):
        """Scale and place object to fit in pane."""
        frame_width = config.frame_width / self.TOTAL_PANES
        scale = (frame_width - self.PADDING * 2) / obj.width
        height_bottom_padding = (self.brace_height  if vert_align == "DOWN" else None)
        self.place_in_pane(obj, pane_index, padding=self.PADDING, height_bottom_padding=height_bottom_padding, 
                          vert_align=vert_align, animate=True, scale=scale)

    def _animate_cta_split(self):
        """Animate CTA split stage."""
        anims, rects, labels = self._create_labels_and_rects(
            self.tensor.ctas, "CTA", 24, "block_id"
        )
        self.play(*anims)
        self.wait(2)
        self._fade_out_all(rects, labels)
        self.wait(1)

    def _animate_warp_split(self):
        """Animate warp split stage."""
        # Remove braces for main tensor
        self._hide_brace_labels(self.tensor)

        self.highlight_code_line(3)
        org_warp = self.tensor.ctas[0].warps[0]
        anims, warp_rects, _ = self._create_labels_and_rects(
            self.tensor.ctas[0].warps, "Warp", 24, "warp_id"
        )
        self.play(*anims)

        # Add and show braces for warps
        warps_rows, warps_cols = self.layout_config.warps_per_cta
        self._add_brace_labels(self.tensor, warps_rows, warps_cols)
        self._show_brace_labels(self.tensor)

        self.wait(2)

        # Remove braces
        self._hide_brace_labels(self.tensor)

        warp_copy = org_warp.copy()
        self.highlight_code_line(2)
        self._highlight_and_focus(org_warp)
        self._scale_to_pane(warp_copy, 1)
        self.add_title("Threads per warp", warp_copy)
 
        return warp_copy

    def _animate_thread_split(self, warp_copy):
        """Animate thread split stage."""
        anims, thread_rects, _ = self._create_labels_and_rects(
            warp_copy.threads, "Thread", 12, "lane_id"
        )
        self.play(*anims)
        
        # Add and show braces for threads
        thread_rows, thread_cols = self.layout_config.threads_per_warp
        self._add_brace_labels(warp_copy, thread_rows, thread_cols)
        self._show_brace_labels(warp_copy)

        self.wait(2)

        # Remove braces
        self._hide_brace_labels(warp_copy)

        self.highlight_code_line(1)

        org_thread = warp_copy.threads[0]
        thread_copy = org_thread.copy()
        
        self._highlight_and_focus(org_thread)
        self._scale_to_pane(thread_copy, 2)
        self.add_title("Thread registers", thread_copy)
        
        # Add and show braces for thread registers
        registers_rows, registers_cols = self.layout_config.size_per_thread
        self._add_brace_labels(thread_copy, registers_rows, registers_cols, 
                              cols_direction=LEFT, cols_shift=(-0.20, 0))
        self._show_brace_labels(thread_copy)

        self.wait(2)

        # Remove braces
        self._hide_brace_labels(thread_copy)

        return thread_copy

    def _highlight_thread_registers(self, warp_copy, thread_copy):
        """Highlight registers for each thread."""
        for i, thread in enumerate(warp_copy.threads):
            if i > 1:  # Limit to first 2 threads
                return
            
            self.play(thread.animate.set_color(ORANGE))

            
            # Collect elements to highlight
            elems = []
            label_anims = []
            clear_anims = []
            
            for r_val, r_elem in zip(thread.registers, thread_copy):
                # Create label
                label = Text(f"{r_val.tensor_index}", font_size=12)
                label.move_to(r_elem.get_center())
                label_anims.append(Write(label))
                clear_anims.append(Uncreate(label))
                
                # Add original tensor element
                x, y = r_val.tensor_index
                elems.append(self.tensor.elements[x][y])
            
            # Add thread elements
            elems.extend(thread)
            
            # Animate highlighting
            self.play(*[e.animate.set_fill(YELLOW, opacity=0.5) for e in elems])
            self.play(*label_anims)
            if i == 1:
                return
            self.wait(2)
            self.play(*clear_anims, run_time=0.1)
            self.play(*[e.animate.set_fill(opacity=0) for e in elems])
            if i == 0:
                self.play(thread.animate.set_color(BLUE))

            self.wait(1)



    def animate_stages(self):
        """Animate all visualization stages."""
        self._animate_cta_split()
        warp_copy = self._animate_warp_split()
        thread_copy = self._animate_thread_split(warp_copy)

        self.unhighlight_code_line()

        self._highlight_thread_registers(warp_copy, thread_copy)

    def construct(self):
        """Main scene construction."""
        self.init()
        self.animate_stages()
        self.wait()

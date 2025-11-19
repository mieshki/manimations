from manim import *
from manim import Scene, VGroup, Text, Tex, MathTex, Code, BraceLabel, MobjectTable, Matrix, IntegerMatrix, Line, Arrow
from manim import BLUE, YELLOW, BLUE_D, GREY, RED, GREEN, WHITE
from manim import LEFT, RIGHT, UP, DOWN
from manim import Create, Uncreate, FocusOn, FadeIn, FadeOut, Write, Unwrite, SurroundingRectangle, LaggedStart, smooth

from gluon_linear_mapping import BlockedLayoutAnalyzer, BlockedLayoutConfig, DeviceConfig
from linear_mapping_decoder import LinearLayoutAnalyzer
import torch
import numpy as np
import re


class LinearMapping2D(Scene):
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

        linear_layout = self.layout_analyzer._linear_layout  
        self.ll_analyzer = LinearLayoutAnalyzer(self.torch_tensor, linear_layout)

    def _calculate_linear_layout_matrix(self, reg_bases_only=False):
        rank = self.ll_analyzer.rank


        reg_bases = self.ll_analyzer.get_reg_bases()
        lane_bases = self.ll_analyzer.get_lane_bases()
        warp_bases = self.ll_analyzer.get_warp_bases()


        if reg_bases_only:
            all_bases = reg_bases
        else:
            all_bases = reg_bases + lane_bases + warp_bases

      
        if not all_bases:  
            return np.array([])  
          
        # Znajdź maksymalną wartość w każdym wymiarze, aby określić liczbę bitów  
        max_vals = [0] * rank  
        for basis in all_bases:  
            for dim in range(rank):  
                max_vals[dim] = max(max_vals[dim], basis[dim])  
          
        # Liczba bitów potrzebna dla każdego wymiaru  
        num_bits_per_dim = [max_vals[dim].bit_length() for dim in range(rank)]  
        total_bits = sum(num_bits_per_dim)  
          
        # Konstruuj macierz A w GF(2)  
        matrix_A = np.zeros((total_bits, len(all_bases)), dtype=int)  
          
        for col_idx, basis in enumerate(all_bases):  
            bit_row = 0  
            for dim in range(rank):  
                value = basis[dim]  
                # Rozłóż wartość na bity  
                for bit_pos in range(num_bits_per_dim[dim]):  
                    if (value >> bit_pos) & 1:  
                        matrix_A[bit_row, col_idx] = 1  
                    bit_row += 1  
          
        return matrix_A, num_bits_per_dim

    def _compute_tensor_index_from_matrix_A(self, matrix_A, reg_id, lane_id, warp_id, block_id, num_bits_per_level, num_bits_per_dim):    
        # Utwórz wektor wejściowy z bitów wszystkich poziomów    
        input_bits = []    
        for level_id, num_bits in zip([reg_id, lane_id, warp_id, block_id], num_bits_per_level):    
            for bit_pos in range(num_bits):    
                input_bits.append((level_id >> bit_pos) & 1)    
            
        # Mnożenie macierzy A × input_vector w GF(2)    
        output_bits = []    
        for row in matrix_A:    
            bit = 0    
            for i, input_bit in enumerate(input_bits):    
                bit ^= (row[i] & input_bit)    
            output_bits.append(bit)    
            
        # Konwertuj bity na numpy array  
        np_output_bits = np.array(output_bits, dtype=np.int8).reshape(-1, 1)  
          
        # Konwertuj bity wyjściowe na logiczne indeksy tensora  
        tensor_indices = []  
        bit_offset = 0  
          
        for num_bits in num_bits_per_dim:  
            # Wyciągnij bity dla tego wymiaru  
            dim_bits = output_bits[bit_offset:bit_offset + num_bits]  
              
            # Konwertuj bity na liczbę dziesiętną (LSB first)  
            dim_index = sum(bit << i for i, bit in enumerate(dim_bits))  
            tensor_indices.append(dim_index)  
              
            bit_offset += num_bits  
          
        return np_output_bits.T, tensor_indices

    def draw_matrix_line(self, m, id0, id1, type="horizontal", color=WHITE,
                     stroke_width=4, size=None):
        entries = m.get_entries()

        if type == "horizontal":
            rows = m.get_rows()
            row1 = rows[id0]
            row2 = rows[id1]

            mid_y = (row1.get_bottom()[1] + row2.get_top()[1]) / 2

            cols = m.get_columns()

            if size is None:
                left_x = entries.get_left()[0]
                right_x = entries.get_right()[0]

            else:
                # left edge of column 0
                left_x = cols[0].get_left()[0]
                # right edge of column (size-1)
                right_x = cols[size - 1].get_right()[0]

            start_point = np.array([left_x, mid_y, 0])
            end_point   = np.array([right_x, mid_y, 0])

        elif type == "vertical":
            cols = m.get_columns()
            col1 = cols[id0]
            col2 = cols[id1]

            mid_x = (col1.get_right()[0] + col2.get_left()[0]) / 2

            rows = m.get_rows()

            if size is None:
                top_y = entries.get_top()[1]
                bottom_y = entries.get_bottom()[1]

            else:
                # top of row0
                top_y = rows[0].get_top()[1]
                # bottom of row(size-1)
                bottom_y = rows[size - 1].get_bottom()[1]

            start_point = np.array([mid_x, top_y, 0])
            end_point   = np.array([mid_x, bottom_y, 0])

        return Line(start_point, end_point, color=color, stroke_width=stroke_width)

    def draw_matrix_symbol(self, m, id0, id1, symbol="\\oplus", color=WHITE):
        rows = m.get_rows()
        row1 = rows[id0]
        row2 = rows[id1]

        # Pozycja Y między wierszami
        mid_y = (row1.get_bottom()[1] + row2.get_top()[1]) / 2

        # Pozycja X – środek macierzy (na podstawie entries)
        entries = m.get_entries()
        mid_x = (entries.get_left()[0] + entries.get_right()[0]) / 2

        pos = np.array([mid_x, mid_y, 0])

        sym = MathTex(symbol, color=color).move_to(pos)

        return sym

    def animate_intro(self):
        #title = Tex(r"Linear Layouts: Efficient Tensor computation using $\mathbb{F}_2$ [Galois Field (2)]", font_size=40, color=BLUE)
        title = Tex(r"Linear Layouts: Robust Code Generator of Efficient \\ Tensor Computation Using $\mathbb{F}_2$ [Galois Field (2)]", font_size=40, color=BLUE)

        self.play(Write(title))

        self.wait(1)
        self.play(FadeOut(title))

    def get_bases_table(self):
        table_data_raw = [
            ["reg_bases", "lane_bases", "warp_bases"],
        ]
        reg_bases = self.ll_analyzer.get_reg_bases()
        lane_bases = self.ll_analyzer.get_lane_bases()
        warp_bases = self.ll_analyzer.get_warp_bases()
        
        max_len = (max(len(warp_bases), len(lane_bases), len(reg_bases)))
        for i in range(max_len):
            reg_base = reg_bases[i] if len(reg_bases) > i else ""
            lane_base = lane_bases[i] if len(lane_bases) > i else ""
            warp_base = warp_bases[i] if len(warp_bases) > i else ""

            table_data_raw.append([f"{reg_base}", f"{lane_base}", f"{warp_base}"])

        table_data_mobjects = []
        for row in table_data_raw:
            table_data_mobjects.append([Text(cell, font_size=30) for cell in row])


        bases_table = MobjectTable(
            table_data_mobjects,
            include_outer_lines=True,
            line_config={"color": GREY},
            h_buff=0.2, # horizontal padding
            v_buff=0.2  # vertical padding
        )
        return bases_table

    def animate_gl_layout_to_linear_layout(self):
        self.current_title = Tex(r"Convert any gluon memory layout to linear layout", font_size=40, color=BLUE)
        self.current_title.to_edge(UP)
        self.play(Write(self.current_title))

        # Display python code for blocked layout to linear layout conversion
        layout_str = self.layout_config.get_layout_code_str()
        bases_str = self.ll_analyzer.get_linear_layout_bases_str()

        code_str = 'blocked_layout=' +  layout_str + '\nlinear_layout = GluonOpBuilder.to_linear_layout(blocked_layout, ...)\n' + bases_str 
        code = Code(
            code_string=code_str,
            language="python",
            background="rectangle",
            add_line_numbers=False,
            background_config={"stroke_color": "maroon"},
        ).scale(0.7).to_edge(DOWN).shift(UP)
        
        for i, line in enumerate(code[1]):
            if i < 6:
                continue
            line.set_opacity(0)

        self.play(Write(code))

        self.wait(2)

        anims = []
        for i, line in enumerate(code[1]):
            if i < 6:
                continue
            line.set_opacity(1)
            anims.append(FadeIn(line))

        self.play(*anims)


        # highlight bases in code block
        ll_bases_code_group = []
        for i, line in enumerate(code[1]):
            if i < 9 or i > 13:
                continue
            ll_bases_code_group.append(line)

        ll_bases_code_group = VGroup(ll_bases_code_group)
        surrounding_rect = SurroundingRectangle(
            ll_bases_code_group,
            color=BLUE_D,
            fill_color=BLUE_D,
            fill_opacity=0.05,
            buff=0.01,
            stroke_width=1
        )
        self.play(Create(surrounding_rect), run_time=1)

        self.wait(4)

        self.bases_table = self.get_bases_table()
        self.bases_table.scale(1.5)

        self.play(Uncreate(surrounding_rect), Uncreate(code))
        self.wait(1)
        self.play(FadeIn(self.bases_table))

        columns = self.bases_table.get_columns()
        anims = []
        # color elements by column
        anims.append(columns[0].animate.set_color(YELLOW))
        anims.append(columns[1].animate.set_color(BLUE))
        anims.append(columns[2].animate.set_color(GREEN))

        self.play(*anims)
        self.play(Unwrite(self.current_title))

    def extract_xy_from_text(self, text_mobj: Text):
        """
        Parse a Manim Text like "[32, 0]" and return (x_int, y_int, x_mobject, y_mobject)
        """
        # Reconstruct string from submobjects
        raw = text_mobj.text
        # Find numbers
        match = re.findall(r"-?\d+", raw)
        if len(match) != 2:
            raise ValueError(f"Expected exactly two integers in '{raw}'")
        x_int, y_int = map(int, match)

        # Locate the first and second number substrings
        num_spans = [m.span() for m in re.finditer(r"-?\d+", raw)]

        def mobj_for_span(span):
            start, end = span
            return VGroup(*text_mobj[start:end])

        x_mobj = mobj_for_span(num_spans[0])
        y_mobj = mobj_for_span(num_spans[1])
        return x_int, y_int, x_mobj, y_mobj

    def animate_reg_bases_matrix_creation(self):
        self.current_title = Tex(r"$\mathbb{F}_2$ matrix creation for registers", font_size=40, color=BLUE)
        self.current_title.to_edge(UP)
        self.play(Write(self.current_title))

        reg_bases = self.ll_analyzer.get_reg_bases()

        # Uncreate lane/warp elements
        reg_rows_count = len(reg_bases)

        copied_bases_table = self.bases_table.copy()
        columns = copied_bases_table.get_columns()

        self.elems_to_uncreate = []
        self.elems_to_uncreate.append(columns[1])
        self.elems_to_uncreate.append(columns[2])
        for e in columns[0][reg_rows_count+1:]:
            self.elems_to_uncreate.append(e)

        self.play(copied_bases_table.animate.set_opacity(1), self.bases_table.animate.set_opacity(0))
        #self.bases_table.set_opacity(0)

        anims = []
        for e in self.elems_to_uncreate:
            anims.append(Uncreate(e))
        self.play(LaggedStart(*anims, lag_ratio=0.1))
        self.wait(1)

        # Move table to right with only reg_bases displayed, hidden cols outside of screen
        hidden_columns_width = columns[1].get_width() + columns[2].get_width()
        self.play(copied_bases_table.animate.to_edge(RIGHT).shift(RIGHT * hidden_columns_width))
        self.play(self.bases_table.animate.to_edge(RIGHT).shift(RIGHT * hidden_columns_width))

        dim0_objs = []
        dim0_vals = []
        dim1_objs = []
        dim1_vals = []
        for t in columns[0][1:]:
            if t.text == "":
                continue
            x,y,xmobj,ymobj = self.extract_xy_from_text(t)
            dim0_objs.append(xmobj)
            dim0_vals.append(x)
            dim1_objs.append(ymobj)
            dim1_vals.append(y)

        dim0_group = VGroup(dim0_objs)
        dim1_group = VGroup(dim1_objs)

        dim0_rect = SurroundingRectangle(dim0_group, color=RED, buff=0.15, stroke_width=6)
        dim1_rect = SurroundingRectangle(dim1_group, color=BLUE, buff=0.15, stroke_width=6)

        # Highlight values for dim0/dim1
        self.play(Create(dim0_rect), Create(dim1_rect))

        dim0_label = Text("dim[0]", font_size=28, color=RED).next_to(dim0_rect.get_bottom()).shift(DOWN * 0.6).shift(LEFT * 1.35)
        dim1_label = Text("dim[1]", font_size=28, color=BLUE).next_to(dim0_rect.get_bottom()).shift(DOWN * 0.6).shift(RIGHT * 0.45)

        dim0_arrow = Arrow(
            start=dim0_rect.get_bottom(),
            end=dim0_label.get_top(),
            color=RED,
            buff=0
        )

        dim1_arrow = Arrow(
            start=dim1_rect.get_bottom(),
            end=dim1_label.get_top(),
            color=BLUE,
            buff=0
        )

        # color table elements:
        anims = []
        for e1, e2 in zip(dim0_group, dim1_group):
            anims.append(e1.animate.set_color(RED))
            anims.append(e2.animate.set_color(BLUE))

        # Create labels with arrows for dim0/dim1
        self.play(Write(dim0_label), Write(dim1_label), Create(dim0_arrow), Create(dim1_arrow), *anims) 


        # Create empty matrix for F2 reg bases
        matrix_A, num_bits_per_dim = self._calculate_linear_layout_matrix(reg_bases_only=True)
        dim_x_height, dim_y_height = num_bits_per_dim

        m0 = Matrix(matrix_A).scale(1)
        for e in m0.get_entries():
            e.set_opacity(0)
        self.play(FadeIn(m0))

        reg_label = Tex("Reg", font_size=40, color=YELLOW)
    
        first_col = m0.get_columns()[0]
        reg_label.next_to(first_col, UP, buff=0.3)
        
        self.play(FadeIn(reg_label)) 

        hor_line = self.draw_matrix_line(m0, 0, dim_x_height, type="horizontal")
        self.play(Create(hor_line))

        objs_array = [[0 for j in range(len(dim0_objs))] for i in range(2)]

        # Move dim[0] vals to matrix
        anims = []
        for i,e in enumerate(dim0_objs):
            t = e.copy()
            dest_cell = m0.get_rows()[0][i]

            anims.append(t.animate.move_to(dest_cell.get_center()))
            objs_array[0][i] = t
        self.play(LaggedStart(*anims, lag_ratio=0.1))

        # Move dim[1] vals to matrix 
        anims = []
        for i,e in enumerate(dim1_objs):
            t = e.copy()
            dest_cell = m0.get_rows()[dim_x_height][i]

            anims.append(t.animate.move_to(dest_cell.get_center()))
            objs_array[1][i] = t
        self.play(LaggedStart(*anims, lag_ratio=0.1))

        self.current_subtitle = Tex(r"Convert to binary", font_size=40, color=BLUE)
        self.current_subtitle.to_edge(DOWN)
        self.play(Write(self.current_subtitle))

        # Display binary translation
        placeholders = [
            Text(r"AA", font_size=35) for _ in range(6)
        ]

        group = VGroup(*placeholders).arrange(
            DOWN, aligned_edge=LEFT, buff=0.25
        )

        group.to_edge(LEFT)
        for e in group:
            e.set_opacity(0)
        self.play(Write(group))

        anims = []
        # dim0 elems
        for i in range(len(m0.get_rows()[0])):
            e = objs_array[0][i]
            anims.append(e.animate.move_to(group[i].get_center()))

        # dim1 elems
        for i in range(len(m0.get_rows()[dim_x_height])):
            e = objs_array[1][i]
            anims.append(e.animate.move_to(group[i+len(m0.get_rows()[0])].get_center()))

        self.play(LaggedStart(*anims, lag_ratio=0.1))

        t = objs_array[0][0]

        def bin_format(x, bit_length):
            return f'0b{x:0{bit_length}b}' 

        anims = []
        txts = []
        for i, o in enumerate(objs_array[0]):
            txt = Text(f"={bin_format(dim0_vals[i], dim_x_height)}", color=RED, font_size=30).scale(1.5)
            txt.next_to(o, RIGHT)
            anims.append(Write(txt))
            txts.append(txt)

        for i, o in enumerate(objs_array[1]):
            txt = Text(f"={bin_format(dim1_vals[i], dim_y_height)}", color=BLUE, font_size=30).scale(1.5)
            txt.next_to(o, RIGHT)
            anims.append(Write(txt))
            txts.append(txt)

        self.play(LaggedStart(*anims, lag_ratio=0.1))

        self.play(Uncreate(self.current_subtitle))

        objs_to_uncreate = []
        objs_to_uncreate_at_end = []

        anims = []
        for i in range(len(objs_array[0])):
            t = txts[i][-1]
            dest = m0.get_rows()[0][i].get_center()
            anims.append(t.animate.move_to(dest))
            objs_to_uncreate.append(txts[i][:-1])
            objs_to_uncreate_at_end.append(t)
        self.play(LaggedStart(*anims, lag_ratio=0.1))

        anims = []
        for i in range(len(objs_array[1])):
            t1 = txts[i+len(objs_array[0])][-1]
            t2 = txts[i+len(objs_array[0])][-2]

            dest1 = m0.get_rows()[1][i].get_center()
            dest2 = m0.get_rows()[2][i].get_center()
            
            anims.append(t1.animate.move_to(dest1))
            anims.append(t2.animate.move_to(dest2))

            objs_to_uncreate.append(txts[i+len(objs_array[0])][:-2])
            objs_to_uncreate_at_end.append(t1)
            objs_to_uncreate_at_end.append(t2)
        self.play(LaggedStart(*anims, lag_ratio=0.1))


        for o in objs_array[0]:
            objs_to_uncreate.append(o)
        for o in objs_array[1]:
            objs_to_uncreate.append(o)

        anims = []
        for o in objs_to_uncreate:
            anims.append(Uncreate(o))

        self.play(*anims)
        self.wait(2)

        anims = []
        anims.append(Uncreate(dim0_arrow))
        anims.append(Uncreate(dim0_arrow))
        anims.append(Uncreate(dim0_label))
        anims.append(Uncreate(dim0_rect))

        anims.append(Uncreate(dim1_arrow))
        anims.append(Uncreate(dim1_arrow))
        anims.append(Uncreate(dim1_label))
        anims.append(Uncreate(dim1_rect))

        anims.append(Uncreate(reg_label))
        anims.append(Uncreate(hor_line))
        for o in objs_to_uncreate_at_end:
            anims.append(Uncreate(o))
        anims.append(Uncreate(m0))
        anims.append(Unwrite(self.current_title))

        self.play(*anims)
        self.temp_table = copied_bases_table

    def animate_full_matrix_creation(self):
        self.current_title = Tex(r"$\mathbb{F}_2$ full matrix creation", font_size=40, color=BLUE)
        self.current_title.to_edge(UP)
        self.play(Write(self.current_title))

        self.bases_table.set_opacity(1)
        self.temp_table.set_opacity(0)
        self.play(
            Uncreate(self.temp_table),
            self.bases_table.animate.scale(0.4).to_edge(RIGHT)
        )

        matrix_A, num_bits_per_dim = self._calculate_linear_layout_matrix()
        dim_x_height, dim_y_height = num_bits_per_dim
        reg_bitwidth = len(self.ll_analyzer.get_reg_bases())
        lane_bitwidth = len(self.ll_analyzer.get_lane_bases())
        warp_bitwidth = len(self.ll_analyzer.get_warp_bases())
        total_bitwidth = reg_bitwidth + lane_bitwidth + warp_bitwidth

        m0 = IntegerMatrix(matrix_A).scale(0.5)
        for e in m0.get_entries():
            e.set_opacity(0)

        m0.to_edge(UP+LEFT).shift(DOWN*1.5)
        self.play(FadeIn(m0))

        lane_anims = []
        hor_line = self.draw_matrix_line(m0, dim_x_height-1, dim_x_height, type="horizontal")

        ver_line1 = self.draw_matrix_line(m0, reg_bitwidth-1, reg_bitwidth, type="vertical")
        ver_line2 = self.draw_matrix_line(m0, reg_bitwidth + lane_bitwidth - 1, reg_bitwidth + lane_bitwidth, type="vertical")

        lane_anims.append(Create(hor_line))
        lane_anims.append(Create(ver_line1))
        lane_anims.append(Create(ver_line2))

        self.play(*lane_anims)

        reg_label = Tex("Register", font_size=40, color=YELLOW)
        reg_col = VGroup(*m0.get_columns()[0:reg_bitwidth])
        reg_label.next_to(reg_col, UP, buff=0.3)

        lane_label = Tex("Thread", font_size=40, color=BLUE)
        lane_col = VGroup(*m0.get_columns()[reg_bitwidth:reg_bitwidth + lane_bitwidth])
        lane_label.next_to(lane_col, UP, buff=0.3)

        warp_label = Tex("Warp", font_size=40, color=GREEN)
        warp_col = VGroup(*m0.get_columns()[reg_bitwidth + lane_bitwidth:reg_bitwidth + lane_bitwidth + warp_bitwidth])
        warp_label.next_to(warp_col, UP, buff=0.3)


        label_anims = []
        label_anims.append(FadeIn(reg_label))
        label_anims.append(FadeIn(lane_label))
        label_anims.append(FadeIn(warp_label))
        
        self.play(*label_anims)


        dim_x_rows = m0.get_rows()[0:dim_x_height]
        dim_y_rows = m0.get_rows()[dim_x_height:]

        dim_x_braces = BraceLabel(
            dim_x_rows, 
            text="dim[0]", 
            font_size=32,
            brace_direction=RIGHT, 
            brace_config={"stroke_width": 0.5},
            buff=0.25
        ).shift(RIGHT * 0.2)

        self.play(Create(dim_x_braces))
        dim_y_braces = BraceLabel(
            dim_y_rows, 
            text="dim[1]", 
            font_size=32,
            brace_direction=RIGHT, 
            brace_config={"stroke_width": 0.5},
            buff=0.25
        ).shift(RIGHT * 0.2)
        self.play(Create(dim_y_braces))

        #Move dim[0] values to matrix
        dim0_objs = []
        columns = self.bases_table.get_columns()
        index = 0
        anims = []
        for column in columns:
            for t in column[1:]:
                if t.text == "":
                    continue
                x,y,xmobj,ymobj = self.extract_xy_from_text(t)
                e = xmobj.copy()
                dim0_objs.append(e)

                m0_row_dim0 = m0.get_rows()[0]
                dest = m0_row_dim0[index].get_center()
                anims.append(e.animate.move_to(dest))
                index += 1
        self.play(LaggedStart(*anims, lag_ratio=0.1))


        # Move dim[1] values to matrix
        dim1_objs = []
        columns = self.bases_table.get_columns()
        index = 0
        anims = []
        for column in columns:
            for t in column[1:]:
                if t.text == "":
                    continue
                x,y,xmobj,ymobj = self.extract_xy_from_text(t)
                e = ymobj.copy()
                dim1_objs.append(e)

                m0_row_dim1 = m0.get_rows()[dim_x_height]
                dest = m0_row_dim1[index].get_center()
                anims.append(e.animate.move_to(dest))
                index += 1
        self.play(LaggedStart(*anims, lag_ratio=0.1))

        # Transform dim[0] values to binary
        anims = []
        for i in range(total_bitwidth):
            obj = dim0_objs[i]

            m0_elems = m0.get_columns()[i][0:dim_x_height]
            anims.append(Uncreate(obj))
            for e in m0_elems:
                anims.append(e.animate.set_opacity(1))

        self.play(LaggedStart(*anims, lag_ratio=0.25, run_time=1))


        # Transform dim[1] values to binary
        anims = []
        for i in range(total_bitwidth):
            obj = dim1_objs[i]

            m0_elems = m0.get_columns()[i][dim_x_height:]
            anims.append(Uncreate(obj))
            for e in m0_elems:
                anims.append(e.animate.set_opacity(1))

        self.play(LaggedStart(*anims, lag_ratio=0.25, run_time=1))

        self.wait(3)

        self.play(Uncreate(self.bases_table), Uncreate(dim_x_braces), Uncreate(dim_y_braces))

        self.current_subtitle = Tex(r"Getting tensor logical index for Register[5] Thread[0] Warp[0]", font_size=40, color=BLUE)
        self.current_subtitle.to_edge(DOWN)
        self.play(Write(self.current_subtitle))


        def create_bit_array(reg_val, thread_val, warp_val):
            """
            Creates a 1D NumPy array of 10 bits based on R, T, W values and widths.
            Widths: REG=3, THREAD=5, WARP=2.
            """
            REG_W = reg_bitwidth
            THREAD_W = lane_bitwidth
            WARP_W = warp_bitwidth

            if reg_val >= (1 << REG_W) or thread_val >= (1 << THREAD_W) or warp_val >= (1 << WARP_W):
                raise ValueError("One of the input values exceeds its allowed bit width.")
                
            final_int = (reg_val << (THREAD_W + WARP_W)) | \
                        (thread_val << WARP_W) | \
                        warp_val


            # Convert the integer to a 10-bit binary string (e.g., '1010000000')
            binary_string = format(final_int, f'0{REG_W + THREAD_W + WARP_W}b')
            
            # Convert the string of characters ('1', '0') into a NumPy array of integers
            bit_array = np.array(list(binary_string), dtype=int)
            
            return bit_array.reshape(1, -1)

        np_input_vector = create_bit_array(5, 0, 0)
        input_vector_font_size = 24
        input_vector = Matrix(np_input_vector, h_buff=0.75).scale(0.5)
        for row in input_vector.get_rows():
            for e in row:
                e.set_opacity(0)

        input_vector.to_edge(UP+RIGHT).shift(DOWN*1.5).shift(LEFT*1)
        self.play(FadeIn(input_vector))


        input_label = Tex(r"Mapping for Register[5] Thread[0] Warp[0]", font_size=26, color=BLUE)
        input_label.next_to(input_vector, UP)        
        self.play(Write(input_label))
        self.wait(2)


        reg_val = 5
        thread_val = 0
        warp_val = 0


        reg_dest = VGroup(*input_vector.get_rows()[0][0:reg_bitwidth])
        thread_dest = VGroup(*input_vector.get_rows()[0][reg_bitwidth:reg_bitwidth + lane_bitwidth])
        warp_dest = VGroup(*input_vector.get_rows()[0][reg_bitwidth + lane_bitwidth:reg_bitwidth + lane_bitwidth + warp_bitwidth])


        reg_item = Text(f'{reg_val}', font_size=input_vector_font_size)
        reg_item.move_to(reg_dest.get_center())

        thread_item = Text(f'{thread_val}', font_size=input_vector_font_size)
        thread_item.move_to(thread_dest.get_center())

        warp_item = Text(f'{warp_val}', font_size=input_vector_font_size)
        warp_item.move_to(warp_dest.get_center())


        self.play(Write(reg_item))
        self.play(Write(thread_item))
        self.play(Write(warp_item))

        # draw vertical lines
        lane_anims = []
        ver_line1 = self.draw_matrix_line(input_vector, reg_bitwidth-1, reg_bitwidth, type="vertical", stroke_width=1)
        ver_line2 = self.draw_matrix_line(input_vector, reg_bitwidth + lane_bitwidth - 1, reg_bitwidth + lane_bitwidth, type="vertical", stroke_width=1)

        lane_anims.append(Create(ver_line1))
        lane_anims.append(Create(ver_line2))
        self.play(*lane_anims)


        self.wait(2)
        anims = []
        anims.append(reg_item.animate.set_opacity(0))
        for e in input_vector.get_rows()[0][0:reg_bitwidth]:
            anims.append(e.animate.set_opacity(1))

        anims.append(thread_item.animate.set_opacity(0))
        for e in input_vector.get_rows()[0][reg_bitwidth:reg_bitwidth+lane_bitwidth]:
            anims.append(e.animate.set_opacity(1))

        anims.append(warp_item.animate.set_opacity(0))
        for e in input_vector.get_rows()[0][reg_bitwidth+lane_bitwidth:reg_bitwidth+lane_bitwidth+warp_bitwidth]:
            anims.append(e.animate.set_opacity(1))

        self.play(LaggedStart(*anims, lag_ratio=0.25, run_time=0.75))


        col_vals = []
        for i, val in enumerate(np_input_vector[0]):
            if val == 1:
                vals = matrix_A[:, [i]]
                col_vals.append(vals)
        
        # copy and transform these cols to horizontal rows on right under input_vector elem
        output_bits, tensor_indices = self._compute_tensor_index_from_matrix_A(  
            matrix_A,
            reg_id=reg_val,  
            lane_id=thread_val,  
            warp_id=warp_val,  
            block_id=-1,  
            num_bits_per_level=[reg_bitwidth, lane_bitwidth, warp_bitwidth, -1],
            num_bits_per_dim=[dim_x_height, dim_y_height]
        )  

        np_calculation_matrix = np.empty((0, total_bitwidth), dtype=np.int32)

        for vals in col_vals:
            np_calculation_matrix = np.vstack([np_calculation_matrix, vals.T])
        np_calculation_matrix = np.vstack([np_calculation_matrix, output_bits])

        calc_matrix = Matrix(np_calculation_matrix, v_buff=1.4, h_buff=0.75).scale(0.5).next_to(input_vector, DOWN)
        for row in calc_matrix.get_rows():
            for e in row:
                e.set_opacity(0)

        self.play(FadeIn(calc_matrix))


        # highlight bit==1 for selected configuration
        # scale and focus per bit
        # then draw rectangle around column
        # then uncreate rectangle and per each element copy it and move_to dest_position
        # with lagged_start and set_opacity to 1 + uncreate for copy
        cols_to_transform = []
        calc_matrix_row_id = 0
        for i, val in enumerate(np_input_vector[0]):
            highlight_anims = []
            anims = []

            if val == 1:
                elem = input_vector.get_rows()[0][i]

                col = m0.get_columns()[i]
                cols_to_transform.append(col)
                # TODO: adequate color per given bit_width reg/thr/wrp
                rect = SurroundingRectangle(
                    col,
                    color=YELLOW,
                    fill_color=YELLOW,
                    fill_opacity=0.05,
                    buff=0.1,
                    stroke_width=2
                )

                scale_factor = 1.5

                highlight_anims.append(FocusOn(elem))

                highlight_anims.append(Create(rect))
                elems1 = []
                elems2 = []
                for col_id, e in enumerate(col):
                    e_copy = e.copy()
                    dest = calc_matrix.get_rows()[calc_matrix_row_id][col_id] 
                    anims.append(e_copy.animate.move_to(dest.get_center()))
                    elems1.append(dest)
                    elems2.append(e_copy)

                self.play(elem.animate.scale(scale_factor), run_time=0.3, rate_func=smooth)
                self.play(*highlight_anims)
                self.play(elem.animate.scale(1/scale_factor), run_time=0.3, rate_func=smooth)
                self.play(LaggedStart(*anims, lag_ratio=0.1))
                calc_matrix_row_id += 1
                self.play(Uncreate(rect))
                anims2 = []
                for e in elems1:
                    anims2.append(e.animate.set_opacity(1))
                for e in elems2:
                    anims2.append(e.animate.set_opacity(0))
                self.play(*anims2)


        self.play(Uncreate(input_vector), calc_matrix.animate.next_to(input_label, DOWN).get_center(), Uncreate(ver_line1), Uncreate(ver_line2))

        # Prepare XOR and = symbols anims
        symbol_anims = []
        for row_id in range(len(calc_matrix.get_rows())):
            if row_id == len(calc_matrix.get_rows()) - 1:
                continue

            symbol = "\\oplus" # XOR symbol
            if row_id == len(calc_matrix.get_rows()) - 2:
                symbol = "="
            symbol = self.draw_matrix_symbol(calc_matrix, row_id, row_id+1, symbol=symbol)
            symbol_anims.append(Write(symbol))


        # Draw XOR symbol(s) anim
        self.play(*symbol_anims[:-1])


        anims = []
        # Draw = symbol anim with results
        anims.append(*symbol_anims[-1:])
        for e in calc_matrix.get_rows()[-1]:
            anims.append(e.animate.set_opacity(1))

        self.play(LaggedStart(*anims, lag_ratio=0.1))

        lsb_label = Tex("LSB", font_size=20, color=WHITE)
        msb_label = Tex("MSB", font_size=20, color=WHITE)
    
        last_row = calc_matrix.get_rows()[-1]

        lsb_label.next_to(last_row, DOWN+LEFT, buff=0.3)
        msb_label.next_to(last_row, DOWN+RIGHT, buff=0.3)

        # draw lines on calc_matrix
        dim_line = self.draw_matrix_line(calc_matrix, dim_x_height - 1, dim_x_height, type="vertical")
        self.play(Create(dim_line))
        
        self.play(Write(lsb_label)) 
        self.play(Write(msb_label)) 



        # result will be with LSB...MSB format
        # convert to MSB...LSB
        reversed_output = output_bits[:, ::-1] 

        reversed_result_matrix = Matrix(reversed_output, v_buff=1.4, h_buff=0.75).scale(0.5).next_to(calc_matrix, DOWN).shift(DOWN*0.5)
        for row in reversed_result_matrix.get_rows():
            for e in row:
                e.set_opacity(0)

        self.play(FadeIn(reversed_result_matrix))
        lsb_label2 = Tex("LSB", font_size=20, color=WHITE)
        msb_label2 = Tex("MSB", font_size=20, color=WHITE)
    
        last_row = reversed_result_matrix.get_rows()[-1]

        lsb_label2.next_to(last_row, DOWN+RIGHT, buff=0.3)
        msb_label2.next_to(last_row, DOWN+LEFT, buff=0.3)

        
        self.play(Write(lsb_label2)) 
        self.play(Write(msb_label2)) 



        # animate move to elems in reversed order from calc_matrix last row to reversed_result_matrix
        anims = []

        elems1 = []
        elems2 = []
        for i, e in enumerate(reversed(calc_matrix.get_rows()[-1])):
            e_copy = e.copy()
            dest = reversed_result_matrix.get_rows()[0][i] 
            anims.append(e_copy.animate.move_to(dest.get_center()))

            elems1.append(dest)
            elems2.append(e_copy)

        self.play(LaggedStart(*anims, lag_ratio=0.1))
        self.wait(1)

        anims = []
        for e in elems1:
            anims.append(e.animate.set_opacity(1))
        for e in elems2:
            anims.append(e.animate.set_opacity(0))
        self.play(*anims)

        # draw lines on calc_matrix
        dim_line = self.draw_matrix_line(reversed_result_matrix, dim_y_height - 1, dim_y_height, type="vertical")
        self.play(Create(dim_line))


        # convert binary to 10th base
        dim0_elems = reversed_result_matrix.get_rows()[0][dim_y_height:]
        dim0_brace = BraceLabel(
            dim0_elems, 
            text="dim[0]", 
            font_size=22,
            brace_direction=DOWN, 
            brace_config={"stroke_width": 0.5},
            buff=0.25
        )
        self.play(Create(dim0_brace))

        dim1_elems = reversed_result_matrix.get_rows()[0][0:dim_y_height]
        dim1_brace = BraceLabel(
            dim1_elems, 
            text="dim[1]", 
            font_size=22,
            brace_direction=DOWN, 
            brace_config={"stroke_width": 0.5},
            buff=0.25
        )
        self.play(Create(dim1_brace))
        self.wait(2)
        self.play(Uncreate(dim0_brace), Uncreate(dim1_brace))

        # display tensor logical x,y index
        tensor_x, tensor_y = tensor_indices
        result_label = Tex(f"Warp[0] Thread[0] Reg[5] = tensor[{tensor_x}, {tensor_y}]", color=BLUE, font_size=28)
        result_label.next_to(last_row, DOWN, buff=0.3).shift(DOWN)
        self.play(Write(result_label))


    def construct(self):
        self.init_layout_analyzer()

        self.animate_intro()
        self.animate_gl_layout_to_linear_layout()
        self.animate_reg_bases_matrix_creation()
        self.animate_full_matrix_creation()
        
        self.wait()



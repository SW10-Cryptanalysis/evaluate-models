import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
from easy_logging import EasyFormatter

# Set up logger
handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TableGenerator:
    """Discovers compiled evaluation logs and formats them into a clean LaTeX landscape table."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.output_tex_path = self.base_dir / "results-table.tex"
        
        # Explicit column ordering for the X-Axis table layout
        self.target_lengths = [350, 400, 450, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000]
        # Explicit row ordering for the Y-Axis matrix structure
        self.target_redundancies = [0, 5, 10, 15, 20, 25, 30, 50, 100, 150, 200, 300]

    def read_accum_data(self) -> Dict[str, Dict[Tuple[int, int], str]]:
        """
        Parses all found accum_results.jsonl profiles.
        Returns a nested structure: matrix[model_name][(length, redundancy)] = "mean / median / best"
        """
        matrix = defaultdict(dict)
        target_file = "accum_results.jsonl"
        
        accum_files = list(self.base_dir.rglob(target_file))
        if not accum_files:
            logger.error(f"No summary logs matching '{target_file}' found in {self.base_dir.resolve()}")
            return {}

        logger.info(f"Extracting aggregated lines from {len(accum_files)} matrix files...")

        for file_path in accum_files:
            model_label = file_path.relative_to(self.base_dir).parts[0]
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        
                        length = record.get("cipher_length")
                        redundancy = record.get("redundancy")
                        mean_ser = record.get("mean_ser")
                        median_ser = record.get("median_ser")
                        best_ser = record.get("best_case_ser")
                        
                        if (
                            length is None or redundancy is None or 
                            mean_ser is None or median_ser is None or 
                            best_ser is None
                        ):
                            logger.warning(
                                f"Skipping incomplete record in {file_path.name} "
                                f"(Missing values: L={length}, R={redundancy}, "
                                f"Mean={mean_ser}, Med={median_ser}, Best={best_ser})"
                            )
                            continue

                        matrix[model_label][(int(length), int(redundancy))] = (float(mean_ser), float(median_ser), float(best_ser))
                        
            except Exception as e:
                logger.error(f"Failed parsing data inside {file_path}: {e}")
                
        return matrix

    def generate_tex_string(self, matrix: Dict[str, Dict[Tuple[int, int], Tuple[float, float, float]]]) -> str:
        """
        Assembles data into a portrait multi-page table utilizing longtable.
        Groups Lengths vertically with multirow, automatically filtering out 
        rows where all models contain empty data.
        
        Compares all models per row using RAW unrounded float values: 
        Bolds the true lowest SER metric, underlines the true second lowest.
        Formats numbers to 2 decimal places wrapped in \texttt{}, preserving standard text slashes.
        """
        if not matrix:
            logger.error(f"Unable to generate LaTeX string: No matrix data was resolved from '{self.base_dir}'.")
            return ""

        # Alphabetically sort the architectures to act as our columns
        sorted_models = sorted(matrix.keys())
        num_models = len(sorted_models)
        
        # Format the top horizontal header text
        escaped_models = [m.replace("_", r"\_") for m in sorted_models]
        model_headers = " & ".join([f"\\textbf{{{m}}}" for m in escaped_models])
        
        # Build the dynamic alignment string with explicit ultra-tight 3pt horizontal spacing
        align_str = "ll " + "".join([r"@{\hspace{3pt}}c" for _ in range(num_models)])

        tex = []
        tex.append(r"\begingroup")
        tex.append(r"\tiny")  
        
        # Open the longtable environment
        tex.append(f"\\begin{{longtable}}{{{align_str}}}")
        
        # --- FIRST PAGE HEADER ---
        tex.append(r"\caption{Models Symbol Error Rate Comparison} \\")
        tex.append(r"\label{tab:results-table} \\")
        tex.append(r"\toprule")
        tex.append(f"\\textbf{{$N$}} & \\textbf{{$\mu$}} & {model_headers} \\\\")
        tex.append(r"\midrule")
        tex.append(r"\endfirsthead")
        
        # --- RUNNING HEADERS FOR PAGES 2+ ---
        tex.append(r"\toprule")
        tex.append(f"\\textbf{{$N$}} & \\textbf{{$\mu$}} & {model_headers} \\\\")
        tex.append(r"\midrule")
        tex.append(r"\endhead")
        
        # --- RUNNING FOOTERS ---
        tex.append(r"\midrule")
        tex.append(r"\multicolumn{" + str(num_models + 2) + r"}{r}{\textit{Continued on next page...}} \\")
        tex.append(r"\endfoot")
        
        # --- FINAL FOOTER ---
        tex.append(r"\bottomrule")
        tex.append(r"\endlastfoot")

        # --- DATA GENERATION LOOP WITH ACTIVE COMPARATIVE FORMATTING ---
        for length in self.target_lengths:
            valid_rows_in_block = []
            
            for redundancy in self.target_redundancies:
                row_raw_tuples = []
                has_real_data = False
                
                # Fetch the precision float tuples from the matrix
                for model_name in sorted_models:
                    val_tuple = matrix[model_name].get((length, redundancy), None)
                    if val_tuple is not None:
                        has_real_data = True
                    row_raw_tuples.append(val_tuple)
                
                if not has_real_data:
                    continue

                # Isolate full-precision lists for independent column-wise processing
                means, medians, bests = [], [], []
                for t in row_raw_tuples:
                    if t is not None:
                        means.append(t[0])
                        medians.append(t[1])
                        bests.append(t[2])
                    else:
                        means.append(float('inf'))
                        medians.append(float('inf'))
                        bests.append(float('inf'))

                # Helper to grab precise 1st and 2nd rank targets
                def get_top_two_ranks(pool: List[float]) -> Tuple[float, float]:
                    valid_nums = sorted([v for v in pool if v != float('inf')])
                    if not valid_nums:
                        return float('inf'), float('inf')
                    first = valid_nums[0]
                    second = next((v for v in valid_nums if v > first), float('inf'))
                    return first, second

                # Compute the TRUE mathematical winners using absolute raw accuracy values
                min_mean, sec_mean = get_top_two_ranks(means)
                min_med, sec_med = get_top_two_ranks(medians)
                min_bst, sec_bst = get_top_two_ranks(bests)

                row_cells = []
                for idx, t in enumerate(row_raw_tuples):
                    if t is None:
                        # Ensures the empty row placeholders match the typewriter aesthetic
                        row_cells.append("")
                        continue
                    
                    # Rule processing helper: Compares absolute precision values,
                    # wraps style macros, and applies \texttt{} to the final output number string.
                    def format_metric(num: float, gold: float, silver: float) -> str:
                        display_str = f"{num:.2f}"
                        if num == gold:
                            styled = f"\\textbf{{{display_str}}}"
                        elif num == silver:
                            styled = f"\\underline{{{display_str}}}"
                        else:
                            styled = display_str
                        
                        # --- THE FIX: Wrap the final styled item cleanly inside texttt ---
                        return f"\\texttt{{{styled}}}"

                    f_mean = format_metric(means[idx], min_mean, sec_mean)
                    f_med  = format_metric(medians[idx], min_med, sec_med)
                    f_bst  = format_metric(bests[idx], min_bst, sec_bst)
                    
                    row_cells.append(f"{f_mean} / {f_med} / {f_bst}")
                
                valid_rows_in_block.append((redundancy, row_cells))
            
            if not valid_rows_in_block:
                logger.info(f"Skipping entire length block N={length} because all rows are empty.")
                continue
                
            # Step 3: Write out protected block rows
            tex.append(f"% --- Protected Length Block: {length} ---")
            dynamic_span_count = len(valid_rows_in_block)
            
            for row_idx, (redundancy, row_cells) in enumerate(valid_rows_in_block):
                row_data_str = " & ".join(row_cells)
                
                if row_idx == 0:
                    prefix = f"\\multirow{{{dynamic_span_count}}}{{*}}{{{length}}}"
                else:
                    prefix = ""
                
                is_last_row = (row_idx == dynamic_span_count - 1)
                row_ending = r"\\" if is_last_row else r"\\*"
                
                tex.append(f"{prefix:<25} & {redundancy:<10} & {row_data_str} {row_ending}")
            
            tex.append(r"\nopagebreak")
            tex.append(r"\cmidrule(lr){1-2} \cmidrule(lr){2-" + str(num_models + 2) + "}")
                
        if tex[-1].startswith(r"\cmidrule") or tex[-1].startswith(r"\nopagebreak"):
            tex.pop()

        tex.append(r"\end{longtable}")
        
        tex.append(r"\begin{flushleft}")
        tex.append(r"\scriptsize \textit{Note:} Data cells are formatted as \textbf{Mean SER / Median SER / Best-Case SER}. ")
        tex.append(r"\end{flushleft}")
        tex.append(r"\endgroup")

        return "\n".join(tex)
    
    def run(self) -> None:
        """Orchestrates reading the data logs and exporting the unified TeX asset file."""
        if not self.base_dir.exists() or not self.base_dir.is_dir():
            logger.error(f"Execution target directory not found: {self.base_dir.resolve()}")
            return

        matrix_data = self.read_accum_data()
        if not matrix_data:
            logger.error("No data matrix resolved. Aborting code generation output.")
            return

        tex_content = self.generate_tex_string(matrix_data)
        
        try:
            with open(self.output_tex_path, "w", encoding="utf-8") as f:
                f.write(tex_content)
            logger.info(f"Successfully generated dynamic LaTeX table file -> {self.output_tex_path.resolve()}")
        except Exception as e:
            logger.error(f"Failed to write output .tex file: {e}")


if __name__ == "__main__":
    # Set up CLI Argument Parser
    parser = argparse.ArgumentParser(
        description="Scans a target directory for summary outputs and outputs a compiled LaTeX sideways results table."
    )
    
    # Strictly required named flag configuration with no default value
    parser.add_argument(
        "--models-dir", 
        required=True, 
        help="The target base directory to search for models."
    )
    
    args = parser.parse_args()

    # Pass the mandatory path argument straight into the generator class
    generator = TableGenerator(base_dir=args.models_dir)
    generator.run()
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

                        cell_value = f"{mean_ser:.2f} / {median_ser:.2f} / {best_ser:.2f}"
                        matrix[model_label][(int(length), int(redundancy))] = cell_value
                        
            except Exception as e:
                logger.error(f"Failed parsing data inside {file_path}: {e}")
                
        return matrix

    def generate_tex_string(self, matrix: Dict[str, Dict[Tuple[int, int], str]]) -> str:
        """Assembles the final text components into a valid, compile-ready LaTeX document fragment."""
        if not matrix:
            logger.error(f"Unable to generate LaTeX string: No matrix data was resolved from '{self.base_dir}'.")
            return ""

        # Generate Header Dynamic Length Alignments
        col_headers = " & ".join([f"\\textbf{{{l}}}" for l in self.target_lengths])
        
        # Total redundancies to span vertically with multirow
        num_redundancies = len(self.target_redundancies)
        
        tex = []
        tex.append(r"\begin{sidewaystable}[p] % Automatically rotates the entire page 90 degrees counterclockwise")
        tex.append(r"\centering")
        tex.append(r"\caption{Comprehensive Summary of Symbol Error Rate (SER) Metrics across Lengths and Redundancies}")
        tex.append(r"\label{tab:results-table}")
        tex.append(r"\setlength{\tabcolsep}{3pt}")
        tex.append(r"\resizebox{\textheight}{!}{% Scales the table perfectly to fit the landscape page height")
        tex.append(r"\begin{tabular}{ll ccccccccccc}") # 11 text coordinate values
        tex.append(r"\toprule")
        tex.append(r"\multirow{2}{*}{\textbf{Architecture}} & \multirow{2}{*}{\textbf{Redundancy}} & \multicolumn{11}{c}{\textbf{Cipher Length ($N$)}} \\")
        tex.append(r"\cmidrule(lr){3-13}")
        tex.append(f"& & {col_headers} \\\\")
        tex.append(r"\midrule")

        # Alphabetically sort the detected models to keep things predictable
        for model_name in sorted(matrix.keys()):
            tex.append(f"\n% --- {model_name.upper()} ---")
            
            # Escape strings like underscores safely for LaTeX rendering stability
            escaped_model_name = model_name.replace("_", r"\_")
            
            # Print row loops per structural design spec block
            for row_idx, redundancy in enumerate(self.target_redundancies):
                row_cells = []
                
                for length in self.target_lengths:
                    val = matrix[model_name].get((length, redundancy), "")
                    row_cells.append(val)
                
                row_data_str = " & ".join(row_cells)
                
                # Dynamic multirow row-spanning size matching the length of your redundancies array (12)
                if row_idx == 0:
                    prefix = f"\\multirow{{{num_redundancies}}}{{*}}{{{escaped_model_name}}}"
                else:
                    prefix = ""
                    
                tex.append(f"{prefix:<40} & R{redundancy:<2} & {row_data_str} \\\\")
            
            tex.append(r"\midrule")
            
        if tex[-1] == r"\midrule":
            tex.pop()

        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}%")
        tex.append(r"}")
        tex.append(r"\begin{flushleft}")
        tex.append(r"\small \textit{Note:} Data cells are formatted as \textbf{Mean SER / Median SER / Best-Case SER}.")
        tex.append(r"\end{flushleft}")
        tex.append(r"\end{sidewaystable}")

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
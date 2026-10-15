"""Compile the COS30082 report from Typst to PDF and verify page count."""
import argparse
from pathlib import Path
import typst
import pypdf

def compile_report(src="reports/report.typ", out="reports/report.pdf", target_pages=4):
    src_path = Path(src)
    out_path = Path(out)
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root_dir = Path(".").resolve()
    typst.compile(str(src_path), output=str(out_path), root=root_dir)
    
    reader = pypdf.PdfReader(str(out_path))
    page_count = len(reader.pages)
    print(f"Report compiled successfully: {out_path} ({page_count} pages)")
    
    if target_pages is not None:
        if page_count != target_pages:
            print(f"WARNING: Target page count is {target_pages}, but generated report has {page_count} pages!")
        else:
            print(f"VERIFIED: Report matches the strict {target_pages}-page constraint exactly.")
    return page_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="reports/report.typ", help="Source Typst file")
    parser.add_argument("--out", default="reports/report.pdf", help="Output PDF file")
    parser.add_argument("--target-pages", type=int, default=4, help="Required page count")
    args = parser.parse_args()
    compile_report(args.src, args.out, args.target_pages)

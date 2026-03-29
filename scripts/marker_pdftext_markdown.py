#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from marker.builders.document import DocumentBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.builders.structure import StructureBuilder
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import save_output
from marker.providers.registry import provider_from_filepath


class PdfTextMarkdownConverter(PdfConverter):
    def build_document(self, filepath: str):
        provider_cls = provider_from_filepath(filepath)
        layout_builder = self.resolve_dependencies(self.layout_builder_class)
        line_builder = self.resolve_dependencies(LineBuilder)
        line_builder.disable_ocr = True
        ocr_builder = self.resolve_dependencies(OcrBuilder)

        document_builder = DocumentBuilder(self.config)
        document_builder.disable_ocr = True

        provider = provider_cls(filepath, self.config)
        document = document_builder(provider, layout_builder, line_builder, ocr_builder)

        structure_builder = self.resolve_dependencies(StructureBuilder)
        structure_builder(document)

        for processor in self.processor_list:
            processor(document)

        return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDFs to Markdown with Marker using PDF text only."
    )
    parser.add_argument("pdfs", nargs="+", help="One or more PDF files to convert.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where converted Markdown folders will be written.",
    )
    parser.add_argument(
        "--extract-images",
        action="store_true",
        help="Also extract images into the output folder.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    config = {}
    if not args.extract_images:
        config["extract_images"] = False
    return config


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    models = create_model_dict()
    converter = PdfTextMarkdownConverter(config=config, artifact_dict=models)

    for pdf in args.pdfs:
        pdf_path = Path(pdf)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF: {pdf}")

        start = time.time()
        rendered = converter(str(pdf_path))
        output_dir = output_root / pdf_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        save_output(rendered, str(output_dir), pdf_path.stem)
        print(f"Saved markdown to {output_dir} in {time.time() - start:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

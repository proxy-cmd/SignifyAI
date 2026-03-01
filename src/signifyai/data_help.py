from __future__ import annotations


def get_data_help_text() -> str:
    return (
        "SignifyAI Data Ingestion (human-friendly)\n"
        "========================================\n"
        "\n"
        "You do NOT need to manually type f_000..f_125 columns.\n"
        "\n"
        "1) Live webcam collection (best for custom signs)\n"
        "   python -u .\\src\\main.py collect --label hello --samples 250\n"
        "   python -u .\\src\\main.py collect --label thanks --samples 250\n"
        "\n"
        "2) Images folder -> CSV\n"
        "   python -u .\\src\\main.py build-image-dataset --images-root .\\data\\raw\\images --out-csv .\\data\\processed\\dataset.csv\n"
        "\n"
        "3) Videos folder -> CSV\n"
        "   python -u .\\src\\main.py build-video-dataset --videos-root .\\data\\raw\\videos --out-csv .\\data\\processed\\dataset.csv\n"
        "\n"
        "4) ZIP URL or local file URL -> images -> CSV\n"
        "   python -u .\\src\\main.py import-url --url https://example.com/signs.zip --out-dir .\\data\\raw\\images\n"
        "   python -u .\\src\\main.py import-url --url file:///D:/datasets/signs.zip --out-dir .\\data\\raw\\images\n"
        "   python -u .\\src\\main.py build-image-dataset --images-root .\\data\\raw\\images --out-csv .\\data\\processed\\dataset.csv\n"
        "\n"
        "After data is ready:\n"
        "   python -u .\\src\\main.py check-dataset --dataset .\\data\\processed\\dataset.csv\n"
        "   python -u .\\src\\main.py train-all --dataset .\\data\\processed\\dataset.csv\n"
    )

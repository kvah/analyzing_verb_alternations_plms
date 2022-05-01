from pathlib import Path
from setuptools import setup


THIS_DIR = Path(__file__).resolve().parent


setup(
    name="alternation_prober",
    version="0.0.1",
    description="Work for LING 575 - Analysing Neural Language Models, Spring, 2022",
    long_description=(THIS_DIR / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/kvah/ling-575-analyzing-nn-group/",
    package_dir={'': 'src'},
    python_requires=">=3.9",
    install_requires=["transformers"],
    entry_points={
        "console_scripts": [
            "get_embeddings=alternation_prober.embeddings.get_embeddings:main",
        ]
    })

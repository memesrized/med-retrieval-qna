from pathlib import Path

from setuptools import find_packages, setup


def get_package_description(filename):
    with open(Path(__file__).parent / filename) as file:
        return file.read()


setup(
    name="mrq",
    package_dir={"": "medretqna"},
    packages=[x for x in find_packages(where="medretqna") if x.startswith("mrq")],
    description="Small medical retrival QnA",
    long_description=get_package_description("README.md"),
    author="Vasilii Salikov",
    license="MIT",
)

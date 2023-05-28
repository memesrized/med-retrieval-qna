from pathlib import Path
from setuptools import setup, find_packages


def get_package_description(filename):
    with open(Path(__file__).parent / filename) as file:
        return file.read()
    
print(find_packages(where="medretqna", include="mrq"))
print([x for x in find_packages(where="medretqna") if x.startswith("mrq")])
# print(find_packages(where="medretqna/src"))
# print(find_packagess(where="medretqna", include="src*"))

# setup(
#     name="mrq",
#     # packages=['medretqna.src'],
#     # packages=['mrq'],
#     package_dir={"mrq" : "medretqna/mrq"},
#     packages=[x for x in find_packages(where="medretqna") if x.startswith("mrq")],
#     description="Small medical retrival QnA",
#     long_description=get_package_description("README.md"),
#     author="Vasilii Salikov",
#     license="MIT",
# )

setup(
    name="mrq",

    package_dir={"" : "medretqna"},
    packages=[x for x in find_packages(where="medretqna") if x.startswith("mrq")],
    description="Small medical retrival QnA",
    long_description=get_package_description("README.md"),
    author="Vasilii Salikov",
    license="MIT",
)
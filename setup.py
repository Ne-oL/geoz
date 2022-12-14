from setuptools import setup, find_packages
import subprocess
import os

geoz_version = (
    subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)

if "-" in geoz_version:
    # when not on tag, git describe outputs: "1.3.3-22-gdf81228"
    # pip has gotten strict with version numbers
    # so change it to: "1.3.3+22.git.gdf81228"
    # See: https://peps.python.org/pep-0440/#local-version-segments
    v,i,s = geoz_version.split("-")
    geoz_version = v + "+" + i + ".git." + s

assert "-" not in geoz_version
assert "." in geoz_version

assert os.path.isfile("geoz/version.py")
with open("geoz/VERSION", "w", encoding="utf-8") as fh:
    fh.write("%s\n" % geoz_version)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='geoz',
    version= geoz_version,
    description='A Library to create Geographic Maps from Unsupervised algorithms',
    py_modules=['geoz'],
    package_dir={'':'src'},
    install_requires=['pandas', 'mlxtend', 'scikit-learn', 'matplotlib', 'geopandas'],
    
    
    extras_require={
        "dev": [
            "pytest >= 3.7",
            "check-manifest",
            "twine"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Khalid ElHaj",
    author_email="KG.Khair@Gmail.com",
    url="https://github.com/Ne-oL/geoz"
    
    )

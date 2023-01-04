from setuptools import setup, find_packages




with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='geoz',
    version='1.5.2',
    license="BSD 3-Clause",
    description='A Library to convert Unsupervised Clustering Results into Geographical Maps',
    py_modules=['geoz'],
    package_dir={'':'src'},
    install_requires=['pandas', 'mlxtend', 'scikit-learn', 'matplotlib', 'geopandas', 'shapely', 'descartes'],
    
    
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
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Khalid ElHaj",
    author_email="KG.Khair@Gmail.com",
    url="https://github.com/Ne-oL/geoz"
    
    )

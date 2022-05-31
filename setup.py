import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fr:
    requirements = fr.read().splitlines()

setuptools.setup(
    name="squeezeformer",
    packages=setuptools.find_packages(include=["src*"]),
    install_requires=requirements,
    extras_require={
        "tf2.5": ["tensorflow>=2.5.0,<2.6", "tensorflow-text>=2.5.0,<2.6", "tensorflow-io>=0.18.0,<0.19"],
        "tf2.5-gpu": ["tensorflow-gpu>=2.5.0,<2.6", "tensorflow-text>=2.5.0,<2.6", "tensorflow-io>=0.18.0,<0.19"]
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires='>=3.6',
)

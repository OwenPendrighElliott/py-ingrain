from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    install_requires=[
        "certifi",
        "pycurl",
        "pydantic>=2.0.0",
        "packaging",
    ],
    tests_require=[
        "pytest",
    ],
    name="ingrain",
    version="0.0.4",
    author="Owen Elliott",
    author_email="none@none.com",
    description="Python client for the ingrain server",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src", exclude=("tests*",)),
    keywords="ingrain embedding triton inference",
    platform="any",
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires=">=3",
    package_dir={"": "src"},
)

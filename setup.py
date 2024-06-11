import setuptools

setuptools.setup(
    name="mik_tools",
    version="0.0.1",
    author='mr-mikmik',
    author_email='oller@umich.edu',
    packages=["mik_tools"],
    url="https://github.com/mr-mikmik/mik_tools",
    description="A collection of tools for working with data and packages",
    install_requires=[
        'numpy',
        'pyyaml',
    ]
)
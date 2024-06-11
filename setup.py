from setuptools import setup,find_packages

setup(
    name="sellmeier",
    version="1.01",
    py_modules=[
        "sellmeier",
    ],
    author="decoherer",
    author_email="63128649+decoherer@users.noreply.github.com",
    description="Calculate index of refraction or poling period for various nonlinear optical materials.",
    url="https://github.com/decoherer/sellmeier",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy','scipy','matplotlib'],
)

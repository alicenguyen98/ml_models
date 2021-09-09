import setuptools
import os

VERSION = '1.0.0'
DESCRIPTION = 'Wrappers for sklearn models'

setuptools.setup(
    name='ml_models',
    version=VERSION,
    author='alicenguyen98',
    author_email='haianh.ng98@gmail.com',
    description=DESCRIPTION,
    packages=setuptools.find_packages(),
    install_requires=['scikit-learn', 'pandas', 'numpy'],
    python_requires=">=3.9",

    keywords=[],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
import setuptools

with open("./README.rst","r") as f:
    long_description = f.read()

setuptools.setup(
  name="DataProTool",
  version="1.2.0-Beta",
  author="Zhang Jiexi",
  author_email="zhangjiexi66696@outlook.com",
  description="It is a library that support advance tools in feature engineering",
  url="https://github.com/Zhang-Jiexi/DataProTool",
  long_description=long_description,
  packages=setuptools.find_packages(),
  install_requires=["numpy >= 1.23.3","pandas >=1.5.0","scikit-learn >= 1.0.2","tqdm >=4.64.1"],
  license='MIT License',
  classifiers=[
  "Programming Language :: Python :: 3.9",
  "License :: OSI Approved :: MIT License",
  "Operating System :: Microsoft",
  "Natural Language :: English",
  "Natural Language :: Chinese (Simplified)",
  "Development Status :: 4 - Beta",
  ],
)
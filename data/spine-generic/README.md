This folder contains two subjects from the spine-generic public database that were used for super-resolution reconstruction. The complete spine-generic dataset is open-source and can be found [here](https://github.com/spine-generic/data-single-subject). The data is originally formatted using the BIDS convention and can be installed locally using `git-annex`. The installation procedure is as follows: 

1. Install git and [git-annex](https://git-annex.branchable.com/install/) (version > 8). That is, upon doing `git-annex version`, it should return a version > 8.0
2. Run the following commands: 

```bash
git clone https://github.com/spine-generic/data-single-subject && \
cd data-single-subject && \
git annex init && \
git annex get .
```

This should download the dataset locally. The size is about 1 GB. If you're only interested in downloading a few subjects, replace the `git annex get .` command with the name of the subject: 
```bash
git annex get $(find . -name "sub-oxfordFmrib")
```

This downloads the both the images and the derivatives. 

 
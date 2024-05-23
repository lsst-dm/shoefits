# ShoeFits

ShoeFits is a prototyping-stage library for serializing arbitrary Python objects to [FITS](https://fits.gsfc.nasa.gov/) files by embedding a JSON tree as a 1-d array, while allowing image data and other arrays to be saved as additional FITS extensions.
Conceptually, it borrows heavily from [ASDF](https://asdf-standard.readthedocs.io/en/1.1.1/intro.html) (and especially [ASDF-in-FITS](https://asdf-standard.readthedocs.io/en/1.1.1/asdf_in_fits.html)), and it aims to have a data model that is as similar as possible to ASDF, but with JSON instead YAML and the embedding in FITS tailored for fast partial reads (even over http).
This allows it to leverage [Pydantic](https://docs.pydantic.dev/latest/) for Python-JSON conversion and schema generation.

Files written by ShoeFits use a narrow subset of FITS, with a focus on allowing general FITS readers to understand most (but not necessarily all) of the file content, particularly images and coordinate transforms with a standard FITS representation.
ShoeFits reads only these files, not arbitrary FITS files written by other code.

ShoeFits is a (tortured, as is standard in astronomy) acronym: Schema-based Hierarchical Object Embedding in the Flexible Image Transport System.

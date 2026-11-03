# Image metadata

The notebook expects `dataset.zip` with the 500 images and two metadata files:

- `hagrid_meta.csv`: the 250 HaGRID crops (label, source, group = anonymized HaGRID user identifier, `original` = identifier of the source photograph in the HaGRID sample);
- `own_meta_final.csv`: the 250 own photographs (label, source, group = shooting situation, original file name, capture time).

The own photographs show the author, in some cases with his face in the background, so the images are not published here. They are available from the author on request.
The HaGRID crops can be rebuilt from the public HaGRID sample with the identifiers in `hagrid_meta.csv`.

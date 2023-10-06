# hafdh-nadhar

Package Python that blur human representations from an image


### Installation

```sh
pdm install --dev
```

### Test-it

```sh
pdm run python tests/example.py
```

### Build and publish

```sh
pdm publish --password <API_TOKEN>
```

### Usage (from the client side)

```py
from hafdh_nadhar.hafdh import hafdh_img
hafdh_img(img_path="/path/to/img.jpg")
```

### Exemple

Before:

![Before](tests/images/boy.jpg)

After:

![After](tests/images/boy-result.jpg)
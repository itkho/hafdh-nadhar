# hafdh-nadhar

Package Python that blur human representations from an image


### Installation

```sh
pdm install --dev
```

### Build and publish

```sh
pdm publish --password <API_TOKEN>
```

### Usage 

#### From Python code

```py
from hafdh_nadhar.hafdh import hafdh_img

hafdh_img(img_path="/path/to/img.jpg")
```

#### From CLI

```sh
hafdh "/path/to/img.jpg"
```

### Exemple

Before:

![Before](https://raw.githubusercontent.com/itkho/hafdh-nadhar/master/tests/images/boy.jpg)

After:

![After](https://raw.githubusercontent.com/itkho/hafdh-nadhar/master/tests/images/boy-result.jpg)
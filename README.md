# Computer Graphics – Text-to-Image

> **To get started:** Clone this repository using
> 
>     git clone http://github.com/alecjacobson/computer-graphics-text-to-image.git
>

## Background

## Tasks

### `bias.zip`

Probe the image generator to identify a **four different**<a href=#different>\*</a> biases backed up with
images and statistics. 

Include one sentence explaining the bias and another explaining its potential
harm.

```
Roughly 40-50% of medical doctors in Canada are female. However, using the prompt "a canadian medical doctor" to generate 20 images, I perceive only 25% to convey a female doctor.

These images reenforce stereotypes that highly paid medical doctors are male. 
```

![](a-canadian-medical-doctor.jpg)

> Notice any other biases exhibited in these images?

<a id=different>\*</a>By "different" I mean both the context and the demographic. So, for example,
"medical doctor" could only appear once and "perceived gender/race" could only
appear once.


### `movie-poster.zip`


Generate at least four movie poster alternatives for one particular real upcoming film. Your posters must:

 - be visually attractive,
 - reflect the [genre](https://en.wikipedia.org/wiki/Film_genre) of the film,
 - fulfill the specific `constraints` listed below (note: you don't need to 
   use the exact text of the constraint in your prompt), and 
 - attempt to include the title of the film as text in the generated image

Note, these constraints were determined by Alec Jacobson without any precise
knowledge of what these films are actually about. Any spoilers are purely
coincidental.


![](the-color-purple.jpg)

> B- Work.


```
"title": "The Color Purple",
"genre": "drama",
"constraints": [
  "two African American women",
  "one woman is adolescent, one is adult",
  "the adult is wearing a straw hat",
  "a southern-style house in the distance"
```

![](night-swim.jpg)

> B- Work. 

```
"title": "Night Swim",
"genre": "horror",
"constraints": [
  "two young adults",
  "one has a beard",
  "a swimming pool",
  "a supernatural element"
  ]
```

![](aquaman-and-the-lost-kingdom.jpg)

> B- Work. 

```
"title": "Aquaman and the Lost Kingdom",
"genre": "superhero",
"constraints": [
  "one aquaman",
  "aquaman has long hair",
  "aquaman is holding a golden trident",
  "aquaman is swimming underwater",
  "an underwater city in the distance"
  ]
```

![](migration.jpg)

> C+ Work. Can you even count?

```
"title": "Migration",
"genre": "computer-animated",
"constraints": [
  "five ducks",
  "at least one is angry"
  "big eyeballs"
  "blue sky background"
  ]
```

![](dashing-through-the-snow.jpg)

> B- Work.

```
"title": "Dashing Through The Snow",
"genre": "comedy",
"constraints": [
  "Police officer",
  "Santa Claus with dark skin color",
  "they are smiling at each other",
  "skyline in the distance"
  ]
```

### `open-ended.zip`

Report on the process of creating a:

 - self portrait,
 - photograph of a home you'd like to live in, or.
 - image of a memory you have.

![](self-portrait.jpg)

 1. Start with a simple 3-4 word prompt.
 2. In one sentence, describe what is correct about the result.
 3. In another sentence, describe what is incorrect about the result.
 4. Augment or edit the prompt to improve the result.
 5. Repeat steps 2-4 until you have tried at least 20 images.


### `story.zip`

Generate a 6-image short story of a character. The character 
must be consistently portrayed. The character cannot be an existing pop culture
entity (e.g., Naomi Osaka, Spiderman, Justin Trudeau). Images must be in a
consistent non-photographic style.

![](cartoon-elephant.jpg)

> B- work. Style is not very consistent.

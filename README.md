# DCGAN-CelebA-tensorflow

Tensorflow implimentation of the [DCGAN](https://arxiv.org/abs/1511.06434)(Deep Convolutional Generative Adversarial Networks) model. Insipired by the official tensorflow DCGAN tutorial and the book Generative Deep Learning and its [github repository](https://github.com/davidADSP/GDL_code).

![](images/Epoch%20120.png)

Image generated at 120 epoch. 
## Paper Features
- Replace any pooling layers with strided convolutions (discriminator) and deconvolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.

This implimentation of the DCGAN paper is based on the rules described above. 
But the following features varies from the original implimentation(although customizable).
- Learning rate is 10^-6 instead of 10^-4, the learning rate turned out too big and resulted mode collapse.
- Instead of the dataset proposed in the paper for facial image generation, the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) was utilized. 
- 
## Model Training

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Requirements

Install all requirements inculding Python 3.x installed.

```
!pip install -r requirements.txt
```



## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

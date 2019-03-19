# SalGAN for Saliency Prediction
Generative adversarial networks (GANs) are generally used to synthesize images which have realistic data distribution. A conventional GAN model consists of two competing networks namely, a generator and a discriminator. The jobs of these networks are exactly as their name suggests.
The generator generates samples whose data distribution is the same as that of the training dataset. The discriminator discriminates between the sample synthesized by the generator and the real sample drawn from the training
dataset. The training of the GAN models proceeds by training the discriminator and the generator alternatively.

The idea of using GANs for saliency prediction has few challenges of its own. These challenges are listed below:
1. In traditional GANs, the input to the generator is some random noise
and it tries to generate realistic images. In case of SalGAN, the input
to the generator is an image and it must learn to generate a realistic
saliency map.
2. SalGAN desires the generated saliency map must correspond to the
input image. Hence, SalGAN provides both the image as well as
the generated saliency map as input to the discriminator. Traditional
GANs does not have such requirements and only provide the generated
images as input to the discriminator.
3. Traditional GANs does not have any ground truth to compare its gener-
ated images; however, in the case of SalGAN, the ground truth saliency
maps are accessible for comparison.

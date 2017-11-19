# hackatum2017-R-D-Challenge
## Inspiration

Convolutional Neural Networks (CNNs) have been the state of the art in computer vision at least since Krizhevsky et al's AlexNet from 2012. The combination of the CNNs plus Max Pooling has dominated (in various tweaked forms) most academic and industrial uses of computer vision ever since. But CNNs suffer from some severe weaknesses, such as being confused by even simple changes to images such as flipping it upside down. Especially the Max Pooling operation, which is used to improve transformation invariance in CNNs, has come under fire, so far so that the "godfather of deep learning" Geoffrey Hinton has said: "The pooling operation used in convolutional neural networks is a big mistake and the fact that it works so well is a disaster."

As such, after years of research, Hinton and colleagues have publishes their work on developing a better alternative to CNNs with Max Pooling: Capsule Networks. Inspired by their clever design (and possibly even biological plausibility), I was inspired to attempt to implement this technique and use it to solve the R&S (and Microsoft AI) challenge at HackaTUM 2017. Since the technique is so new, I don't yet know how good it will work or not, and am excited to see the results, even if they aren't great.

## What it does

This code implements Capsule Networks, as described in the paper "Dynamic Routing Between Capsules" by Sabour et al (https://arxiv.org/pdf/1710.09829.pdf), modified to apply to the R&S challenge of classifying logos.

Currently, it sadly does nothing practically useful, at least not on a feasible machine of the current day. In theory, it can classify logos in pictures, but due to the problems I ran into (as detailed below), I can't really demonstrate that. It turns out that getting proper results out of this technique is currently just not really computationally feasible (at least as I have implemented the technique). In a way, the most interesting part of this project might be the scientific value in empirically seeing how applying Capsule Networks to a real life problem works, or (in this case) doesn't.

## How I built it

With a lot of determination and caffeine. First, as any good data scientist, I took a good hard look at my data, looking at how it's formatted, what features seem the most relevant and any possible pitfalls I might run into. Using PyTorch, I started by implementing a CapsuleLayer module that can be plugged into any other architecture relatively easily. I then started work on code for loading the dataset and running training jobs. The code was written to (in theory) allow easy switching out of the model architecture, including loss functions and preprocessing operations. 

Once the framework code and the first model architecture (which I lovingly call "Politically Correct", because it uses greyscale images and therefor doesn't see color) was finished, it was time to upload it into the cloud and start debugging and training. Unfortunately, this is where my misadventures begin.

Things started with the very disappointing revelation that the Azure passes provided by Microsoft do not actually allow the renting of GPU instances, which are basically mandatory to train any decent neural network nowadays. Not wanting my project to fail because of this, I started asking around if people had ideas. To my delight, people from the HackerTUM staff and the companies I talked to alike were very helpful and understanding.

I first talked with a staff member about the possibility of using the LRZ supercomputer, but it was booked for experiments (and my code is unlikely to have run on it anyways). Then we considered installing an IPython notebook on his own rig, though this wasn't very fitting for what my code did either. We finally came up with trying the trial version of the Google Cloud, which offers 300$ of free computing. Signing up worked quickly and easily and it seemed like I was ready to go, I just had to gnaw my way through some documentation to get to those GPUs.

But it was not meant to be so. Turns out that, as written so deeply hidden in the documentation you could think they were trying to hide it, Google Cloud too does not offer GPU instances to trial users. Though it DOES provide GPUs under their ML Engine service. Just one problem: To use it, your model must be implemented in TensorFlow, which is quite unfortunate as mine was implemented in PyTorch. Another idea I came to at the time was that I could go home and try to run my code on my desktop there, though it was a 2 hour trip one way, and I wanted to stay and continue working here. So, I was put in front of the decision of losing 4 hours to travel time home and back, or rewriting my entire program in TensorFlow.

At this point in my journey I talked with the lovely people from R&S, and they offered me the option of creating an account on Amazon Web Services, and refunding me the money of renting a GPU instance there. Awesome! Luckily a friend was there to lend me his credit card so we could create an AWS account. Everything is set up, we're ready to go, click on create a new VM aaaand..."Your account needs to be activated, this could take up to 24 hours." I think I can be forgiven for thinking the universe was playing a prank on me at this point.

So I was back to going home or rewriting my code completely. After a minor nerve-induced break, I pulled myself together and decided that I would not let any of this beat me, I'd already come too far, and I will completely recode everything in TF! Sleep is for the weak!

But twists remain on my path. Just as I am about to start the tedious task of rewriting, I am messaged by a staff member that had managed to organize an Azure access that would allow me to use GPU instances! Needless to say, it didn't work right away as it should. After some troubleshooting, I am put into contact with yet another person who had found that, using the command line interface, renting GPU instances worked, but not over the web interface. Because coding is hard, even for Microsoft. 

And so another hour or two go down the drain as I familiarize myself with the arcane documentation of Azure (though I must mention it's not nearly as arcane as Google's) and how to configure a VM to my needs. Eventually though, I get things running and I can finally start debugging my code properly.

And what a debugging process it was! There's a saying that "Debugging is twice as hard as coding, so if you code something as clever as you can, you won't be smart enough to debug it", and I feel like I came face to face with this saying a few times. If one were to read the commit log of my github repo I think one could probably see my slow mental unraveling as tiredness really started taking its toll. 

Finally, tired out of my mind, having gone through iteration after iteration of my code, I come to the dawning realization I expected, a computer science Murphy's Law if one will: My code works, but there's no computer in the world powerful enough to actually run it. Awesome.

I tweak and redo, change parameters and configurations, trying somehow to make it work, but no chance. Eventually, I just decide to get things running, even if the problem is scaled down to such a degree it becomes meaningless. Scaling down the problem more and more, I get to a point where the code can actually run...which promptly unmasks a new, insidious bug hiding in my implementation.

Staring at it for a long time, I feel the creeping temptation of just giving up. Solving this problem involved some serious linear algebra, and my tired brain was simply not equipped to do something like that at that time. But as luck would have it, just as I am ready to give up, something clicks in my head. Some furious, movie-hacker style typing later and voila! My code is running! Not that it's doing anything particularly useful, but it IS running and god damn it that's an accomplishment for me!

Somehow despite all the frustrating problems, I kept a good mood throughout the entire ordeal. Maybe it was the joy of working on a challenging problem, maybe it was the excitement of trying something new and experimental, finding new knowledge, maybe it was the simple fact I got to spend time with my friends. But whatever it was, I had a blast, and I don't regret any of it, even if my results are a little disappointing from a practical standpoint. But I do think there is scientific merit here, in having tried out a new, interesting algorithm and found that it definitely still has some improving to do.

## Accomplishments that I'm proud of

* Not giving up
* Getting something to work, even if trivial

## What I learned

* A LOT about cloud computing and PyTorch
* CapsuleNets are cool but in their current state barely feasible
* Google documentation is my personal hell
* Anything can be fun, as long as you do it with friends

## What's next for HackaTUM Rohde & Schwarz Challenge using CapsuleNets

While the R&S challenge will be over, I definitely am excited to continue experimenting with CapsuleNets. Only recently another interesting paper on the topic, "Matrix Capsules with EM Routing", has been published, which is filled with exciting improvements to the algorithms involved that I'd love to experiment with. I'd like to get my implementation properly cleaned up, not leave it in the mostly undocumented, messy state it is now, and try different designs for the architecture of the model. Whether I will have the time between Uni work to do this any time soon I don't know, but I am pretty sure this won't be the last time I play with CapsuleNets.



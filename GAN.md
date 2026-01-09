Initially we set up all the important libraries required to us , like torch torchaudio torchvision transformers etc... After that we need to mount drive and then set path of audio samples 

Now the real work starts---->

# AUDIO DATASET CLASS
After importing the libraries , first it is important to create an Audio dataset class , where initially we are creating a constroctor called init and gave it some paramets life max_frames , fraction , samplerate etc.. and called them afterwards . We index mapped the class
 --- self.class_to_idx = {cat: i for i, cat in enumerate(categories)} --- > convert string to no.s

 After that we ar looping over the categories and then collects all the audio files.

 Now when we are sampling the files , here the fracntion sample is useful ,
  num_to_sample = int(len(files_in_cat) * fraction)  ---> fraction is set as 1 , hence all the files will be selected , if eg 0.7 was set as frac, randomly any 70% of files would have been selected ,which obviously helps when we do not need to load all the files , and a list of loaded file is then created.

  def __len__(self):
        return len(self.valid_files)  --- > is req to know that how many items are ready to be iterated
      
_getitem_ helps in accessing the files ---- > dunder method

signal -- > tensor of shape [channel , time ] - ie [1,16000]
1 --- > mono 2 ---> stereo  16000 --- > i sec if 32000----> 2 sec.
After that we neeed to make 16k Hz uniform in every audio input , hence we need to resample it . If needed we also can convery stereo to mono.
Now this is pretty important . We need to create spectograms .
n_fft --> we need to analyze chunks of 1024 samples
sr --- > 16000 ie 1024/16000 = 0.064s
hop length -- > controls how far a window can move . 256 means each frame starts with 256 samples after a new one.
n_mels -- >  mel bands compress to spectrum ( ie human perceived freq)

Now we need to make spect. uniform , hence we can eirhter pad or trim them

            _, _, n_frames = log_spec.shape
            if n_frames < self.max_frames:
                pad = self.max_frames - n_frames
                # Pad along the time dimension (last dimension)
                log_spec = F.pad(log_spec, (0, pad))


label_vec = F.one_hot(torch.tensor(label), num_classes=len(self.categories)).float()   ----> OH labeling categories


# CREATING GENERATOR CLASS

Generator inherits nn.module --- takes noise

self.fc = nn.Linear(latent_dim + num_categories, 256 * 8 * 32)

noise vector of latent_dim  , OH encoded num_categroy . latent+category = single vector.
This vector is projected (via Linear) into a large tensor that can be reshaped into a small “image-like” feature map of shape (256, 8, 32).

nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 16x64

            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU() 

----- > this is a deconvolutional laters /upsampling layers which increase with each step from (8×32) up to (128×512),ConvTranspose2d doubles the height and width .
BatchNorm2d ---- >Uses CNN to stabilize training.When generator applies transposed convoltion , the val inside feature map can vary , batch norm feature normalizes them , mean =0 variance = 1.
ReLu ---> introduces non linearity and ensures positive outputs.

def forward .....
----> z+y concatenation

h = self.fc(h)

h = h.view(-1, 256, 8, 32)   ---> turns flat vec into 3D
And the fake spec will be transf to discriminator for training


# CREATING DISCRIMINATOR CLASS

We need to properly differentiate what Generator and Discriminator are doing to understand what is going on with the code.

Generator creates fake spectograms from noise and labels , whereas discriminator determines it.

Generator spect shows (1, 128, 512) , discri outputs A single scalar/logit per sample (real = 1, fake = 0) ,

generator upsamples (convtrasnpose2d) , whereas discr downsamples (conv2d) ----> most imp

generator uses ReLu(+ve) , discr uses LeakyReLu (smol slope of -ve )


# # UTILITY FUNCTIONS (GENERATION, SAVING)

More the sample rate , better the quality of catching harmonics , but lesser files computes
z --> numsamples , latent dim


# 4. GAN TRAINING FUNCTION

We have one optimizer to each of the setups  G/D .
Beta 1 = 0.5 Beta 2 = 0.999

B1 = CONTROLS HOW MUCH MOMENTUM IS KEPT FOR GRADIENTS. Higher the B1 , more the smoothness , slower the rkn to rapid changes
B2 = MOMENTUM FOR SQUARED GRADIENTS .

BCEwithlogits -- Binary Cross-Entropy + logits (discriminator output is NOT passed through sigmoid manually).

tqdm -- > display progress bar of each epoch


  
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_specs = generator(z, labels)
We need to generate fake noise samples as well

.detach()  ---> helps not being updated in the same step for discr . But when training generator , we will not wirte .detach() , as we want to pass fake data WITHOUT .detach() - gradients flow to generator.  

Then we need to update generator weights and track the losses.

We will then plot spectograms and egenrate sample audio (specto --> audio)


    torch.backends.cudnn.benchmark = True
CUDNN chooses the best kernels , which is constant in size .not reproducible if sizes vary.
    USE_AMP =True
enables mixed precisions to speed up training

BCE ---> Binary crossentropies on prob  ---> gives numerical stability .
Then adam optimizer was seperately applied on Generaor and discri.

scalar = .... is used to prevent underflow.(ie rounding of a very small number to zero)

    (batch_size ,1)  
---> discriminator can only give 1 pred per sample

While training Discr , we need to code about fake and real losses and while training Genr , we need to create new fake data as fake data used in discr is detached [detach()] and generator and discriminator should by independent of each other as Genr is trying to create fake audio samples an Discr is trying to identify fake and real samples.

    if (epoch + 1) % 1 == 0: 
-----> This means that after every epoch is loaded , Audio sample and spectogram img is produced . 

    









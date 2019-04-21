	<!-- About Section -->
	<div class="w3-container" style="padding:128px 16px">
		<div class="w3-center">
			<img src="README_resources/images/title.png" alt="title">
			<div class="w3-xxlarge">ABOUT THE PROJECT.</div>
		</div>
		<p class="w3-center w3-large">
			In this section, I will talk about how my model works.
		</p>
	</div>
	
	
	
	
	
	<!-- Detail Section -->
	<div class="w3-container w3-light-grey" style="padding:128px 10%">
		<h3 class="w3-center" style="margin-bottom:32px">Before we talk about music composition AI</h3>
		<div class="w3-center">
			<img src="README_resources/images/about/rnn-lstm-cell.png" id="img_lstm" style="width:60%" alt="title"> </br>
		</div>
		
		<p class="w3-center w3-large" style="margin:32px">
			Currently existing music composition AI are all comprised using RNN or LSTM network.
			But can we really call it a "composition", if what it does is merely  predicting the next data from the data tha came before?
			I don't think those music that were composed using RNN or LSTM networks are legitimate music.
			Composers do not make the next note by looking at the previous note.
			They don't write note 'Sol' because 'Do' 'Re' 'Mi' 'Fa' comes before it.
			Let's consider an example from the artist's perspective.
			Van Gogh doesn't paint blue at the point (0,2) becaue there is red at (0,0) and yellow at (0,1).
			Every artist expresses art based on their inspiration.
			So I think, in order to compose "real music", we need to free ourselves from RNN or LSTM networks.
		</p>
		
		<div class="w3-center">
			<img src="README_resources/images/about/composer.png" id="img_composer" style="width:30%" alt="title">
			<img src="README_resources/images/about/vangogh.png" id="img_vangogh" style="width:30%" alt="title"> </br>
		</div>
	</div>





	<!-- Step 1 -->
	<div class="w3-container w3-white" style="padding:128px 10%">
		<h3 class="w3-center w3-xxlarge">Step 1. Process</h3>
		<p class="w3-large" style="margin:32px">
			The following steps are algorithmic representations of artists' creative processes. </br>
			1. Getting inspiration from some objects or events. </br>
			2. Draw the entire outline. </br>
			3. Add and draw details. </br>
			I thought the generative models I'll be using here have highly similar work flow with the above steps.
		</p>
		<div class="w3-center w3-large">
			<img src="README_resources/images/about/process.png" style="width:60%" alt="title"> </br>
			Similarity between artist's creation process and generative model </br>
			<img src="README_resources/images/about/generator.png" id="img_dcganModel" style="width:60%" alt="title"> </br>
			DCGAN, one of the famous generative models </br>
		</div>
	</div>
	
	
	
	
	
	<!-- Step 2 -->
	<div class="w3-container w3-light-grey" style="padding:128px 10%">
		<h3 class="w3-center w3-xxlarge">Step 2. Midi data to network</h3>
		<p class="w3-center w3-large" style="margin:32px">
			I created an AI model using the GAN model, which is the most mentioned since the NIPS started.
			I collected 1024 midi data.
			Midi is one of the formats for expressing music.
			It uses the note and duration in time sequence.
			Raw midi data is not suitable for editing.
			So I refined it to be more easily editable.
			Also I made training data into fixed-length data.
			To improve data loading speed in network virtual environment I assembled data to chunks.
			You can see the results of my project in the work section.
		</p>
		
		<div class="w3-center w3-large">
			<img src="README_resources/images/about/midiData_0.png" style="width:30%" alt="title">
			<img src="README_resources/images/about/midiData_1.png" style="width:30%" alt="title"> </br> </br>
			<img src="README_resources/images/about/midiData_2.png" style="width:30%" alt="title">
			<img src="README_resources/images/about/midiData_3.png" style="width:30%" alt="title"> </br> </br>
			Real data (target data) </br>
		</div>
	</div>





	<!-- Step 3 -->
	<div class="w3-container w3-white" style="padding:128px 10%">
		<h3 class="w3-center w3-xxlarge">Step 3. Model & Training</h3>
		<p class="w3-large" style="margin:32px">
			For this project I've gone through a lot of trials and errors. 
			If you are curious about them, you can read them in the history section.
			I decided to use the DCGAN model which is known as the most proven GAN model.
			The data shape used in the training, However, did not match the input shape of the DCGAN model itself.
			So I changed parameters in convolution network and modified the last layer of discriminator to go through the fully connected layer twice.
			And to generate [64, 256] shape data, I also modified the generator's noise vector to extend from [2, 8].
		</p>
		
		<div class="w3-center w3-large">
			<img src="README_resources/images/about/model_D_arch.png" style="width:50%" alt="title"> </br>
			Architecture of Discriminator </br>
		</div>
		
		<div class="w3-center w3-large">
			<img src="README_resources/images/about/model_G_arch.png" style="width:50%" alt="title"> </br>
			Architecture of Generator </br>
		</div>
		
		<div class="w3-center w3-large">
			<img src="README_resources/images/about/training.png" style="width:80%" alt="title"> </br>
			Training... </br>
		</div>
		
		<div class="w3-center w3-large">
			<img src="README_resources/images/about/training-weight.png" style="width:50%" alt="title"> </br>
			Training : weight of each sequence note</br>
		</div>
	</div>


	
	
	
	<!-- Reference -->
	<div class="w3-container w3-black" style="padding:64px 10%">
		<h3>Reference</h3>
		
		<div class="w3-large" style="margin-top:32px">
			<a href="#img_lstm" class="w3-black anchorNoDeco"> 1) LSTM Architecture image : http://colah.github.io/posts/2015-08-Understanding-LSTMs/ (2018.11.01) </br> </a>
			<a href="#img_composer" class="w3-black anchorNoDeco"> 2) Tchaikovsky faces : http://witsens.tistory.com/74 (2018.11.01) </br> </a>
			<a href="#img_vangogh" class="w3-black anchorNoDeco"> 3) Bedroom in Arles : https://pt.wikipedia.org/wiki/Ficheiro:Vincent_Willem_van_Gogh_135.jpg (2018.11.01) </br> </a>
			<a href="#img_dcganModel" class="w3-black anchorNoDeco"> 4) DCGAN Architecture image : https://arxiv.org/abs/1607.07539 </br> </a>
		</div>
	</div>
	
		<!-- About Section -->
	<div class="w3-container" style="padding:128px 16px">
		<div class="w3-center ">
			<img src="README_resources/images/title.png" alt="title">
			<div class="w3-xxlarge">HISTORY OF THE PROJECT.</div>
		</div>
		<p class="w3-center w3-large">
			I've gone through a lot of trials and errors to complete the project.</br>
			Following logs are the history of the project.
		</p>
	</div>
	
	
	
	
	
	<!-- Motivation -->
	<div class="w3-container w3-light-grey" style="padding:64px 10%">
		<h3 class="w3-center w3-xxlarge">Motivation</h3>
			<p class="w3-center w3-large" style="margin-top:32px">
				Currently existing music composition AI are all comprised using RNN or LSTM network.
				But I don't think those music that were composed using RNN or LSTM networks are legitimate music.
				Meanwhile, the Generative models, especially GAN and VAE, drew my attention. 
				And I found that most of them are used for mostly processing images.
				I thought it would make more interesting result, if I apply this model to compose music. 
				Moreover I, recently, read interesting paper that the author's team successfully generated 3MB image of people's face (High resolution image).
				Then the idea popped up to me, 'If AI model can make 3MB image, then it would be also possible to make 3MB music'.
				So it was the basic idea that started my project.
			</p>
		
	</div>

	
	
	
	
	<!-- First try - VAE -->
	<div class="w3-container w3-white" style="padding:128px 10%">
		<h3 class="w3-center w3-xxlarge">First try - VAE</h3>
		<div class="w3-center w3-large">
			<img src="README_resources/images/history/vae-gaussian.png" id="img_vaeModel" style="width:60%" alt="title"> </br>
		</div>
		
		<p class="w3-center w3-large" style="margin:32px">
			The first AI-generated model I've used is VAE.
			Unfortunately, however, VAE was not able to achieve the desired result.
			And it lead me to postulate the following explanation for the failure of VAE model.
			1. The VAE model itself has a blurring problem.
			It appears in the form of noise in music. 
			To check the degree of noise caused by the VAE model, I tried overfitting for one isolated data. 
			Then when I reconstructed the train data, this resulted in emergence of unwanted noise.
			2. VAE is essentially based on Monte Carlo estimation.
			Unlike image files, mini batch of the wave file that I use failed to represent a large batch. 
			These lead me to the conclusion that VAE model is not suitable for my project.
		</p>

		<div class="w3-center w3-large" style="margin:32px">
			<audio controls preload="none">
				<source src="README_resources/audio/history/VAE1-ReconstructionWave0.mp3" type="audio/mpeg">
				Your browser does not support the audio element.
			</audio></br>
			1 data overfitting - reconstruction </br>
			Turn down the sound before listening to the music. It can be very loud.
		</div>
	</div>

	
	
	
	
	<!-- Second try - FFT -->
	<div class="w3-container w3-light-grey" style="padding:128px 10%">
		<h3 class="w3-center w3-xxlarge">Second try - FFT</h3>
		
		<p class="w3-center w3-large" style="margin:32px">
			After failure of VAE trial, I contemplated how to use the train data for my model efficiently.
			And following my professor's advice, I thought the Fourier transform has a potential to solve my problem.
			The Fourier transform was effective at representing wave and had the ability to compress the wave files.
			So I decided to use Fourier transform for training data.
			And while trying to implement the Fourier transformations, another possible application of this model occurred to me.
			The value of converting the wave file using Fourier's method is represented by the substitution of three values with a real value, a false value, and a frequency value.
			So if I changed the one-dimensional wave sequence into two dimension, then the data can be expressed in three-dimension. 
			It means that Fourier transformation makes it much easier for computer to deal with the wave file as an three dimensional image. 
			(real value, false value, frequency value in Fourier transformation -> RGB values in image)
			And if I could represent music file as image file, I thought it would be possible to use the GAN model which has been extensively researched for image generation.
			Thus, I decided to apply DCGAN model to this AI project.
		</p>
		<div class="w3-center w3-large">
			<img src="README_resources/images/history/fft_01wave.png" style="width:30%" alt="title">
			<img src="README_resources/images/history/fft_02fft.png" style="width:30%" alt="title"> </br>
			1) Wave file to Fourier transformation </br>
			<img src="README_resources/images/history/fft_03RGB.png" style="width:60%" alt="title"> </br>
			2) Transformation to 3 dimensional data </br>
			<img src="README_resources/images/history/fft_04model.png" style="width:60%" alt="title"> </br>
			3) Normalization and applying to model </br>
			<img src="README_resources/images/history/mytarget_0.png" style="width:15%" alt="title">
			<img src="README_resources/images/history/mytarget_1.png" style="width:15%" alt="title">
			<img src="README_resources/images/history/mytarget_2.png" style="width:15%" alt="title">
			<img src="README_resources/images/history/mytarget_3.png" style="width:15%" alt="title"></br>
			4) Above is the target data in AI model
		</div>
	</div>

	
	
	
	
	<!-- Third try - GAN -->
	<div class="w3-container w3-white" style="padding:128px 10%">
		<h3 class="w3-center w3-xxlarge">Third try - GAN</h3>
		
		<div class="w3-center w3-large">
			<img src="README_resources/images/history/gan-model.png" id="img_ganModel" style="width:50%" alt="title"> </br>
			Idea of the GAN</br>
			<img src="README_resources/images/history/dcgan_architecture.png" id="img_dcganModel" style="width:50%" alt="title"> </br>
			DCGAN architecture - Generator
		</div>
		
		<p class="w3-center w3-large" style="margin:32px">
			Although the GAN model is famously known for being difficult to train, I expected the result to be successful.
			This is because the DCGAN model is clearly validated in the image processing field.
			Although there was no guarantee that this attempt would work in music composition, It seemed very feasible.
			I collected 10,240 data and moved its environment to a cloud service that supports graphics card operations.
			To reduce wave file complexity, the sample rate was lowered from 2048 to 1024.
			There has been numerous attempts.
			I tried the DCGAN itself.
			I created and tested 1D DCGAN using 1D convolution network.
			And I also tried making custom network rather than DCGAN architecture.
			Unfortunately, however, all of the attempts didn't work.
			Following images are the result of the model when I borrowed the GAN architecture.
			As you can see, the significant difference from the target data can be shown from the above images.
		</p>

		<div class="w3-center w3-large" style="margin:32px">
			<img src="README_resources/images/history/myResult_0.png" style="width:15%" alt="title">
			<img src="README_resources/images/history/myResult_1.png" style="width:15%" alt="title">
			<img src="README_resources/images/history/myResult_2.png" style="width:15%" alt="title">
			<img src="README_resources/images/history/myResult_3.png" style="width:15%" alt="title"> </br>
			Generated data of DCGAN (Image) </br> </br>
			<audio controls preload="none">
				<source src="README_resources/audio/history/GAN-Generated.mp3" type="audio/mpeg">
				Your browser does not support the audio element.
			</audio></br>
			Generated data of DCGAN (Audio) </br>
			Turn down the sound before listening to the music. It can be very loud.
		</div>
	</div>

	
	
	
	
	<!-- And current project -->
	<div class="w3-container w3-light-grey" style="padding:128px 10%">
		<h3 class="w3-center w3-xxlarge">Current project</h3>
		<p class="w3-center w3-large" style="margin:32px">
			While I tested the model, I suddenly started to question the validity of the music.
			</br></br>
			"If AI model can make similar image then is it really worth for music?".
			</br></br>
			So I tried its feasibility by using a reassembling technique.
			I collected 50 real data and sliced them into 50 pieces and reassembled them.
			The output of this technique is very similar to the target image.
			But it's sound is not good as I expected.
			I also tried Overfitting to reduce complexity of the problem. 
			But generated data, however, was close to null. 
			I suspected the reason for the problem was activation function ReLU in AI model. 
			So I decided to change it to leakyReLU. 
			It didn't work out either.
			I've tried various approaches, but, as you can see, All the outputs has failed to produce meaningful results. 
			So I decided to change training data format to midi.
			That is why I decided to use midi format for music composition AI model.
		</p>
		<div class="w3-center w3-large">
			<img src="README_resources/images/history/fft_reassemble.png" style="width:50%" alt="title"> </br>
			1) Result : Reassemble </br>
			<img src="README_resources/images/history/gan_overfitting.png" style="width:50%" alt="title"> </br>
			2) Result : Overfitting
		</div>
	</div>
	
	
	
	
	
	<!-- Reference -->
	<div class="w3-container w3-black" style="padding:64px 10%">
		<h3>Reference</h3>
		
		<div class="w3-large" style="margin-top:32px">
			<a href="#img_vaeModel" class="w3-black anchorNoDeco"> 1) VAE model image : https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html (2018.11.01) </br> </a>
			<a href="#img_ganModel" class="w3-black anchorNoDeco"> 2) GAN model image : https://towardsdatascience.com/understanding-generative-adversarial-networks-4dafc963f2ef (2018.11.01) </br> </a>
			<a href="#img_dcganModel" class="w3-black anchorNoDeco"> 3) DCGAN Architecture image : https://arxiv.org/abs/1607.07539 </br> </a>
		</div>
	</div>
	
	
	
	
	
		<script src="README_resources/js/showSpin.js"> </script>






	<!-- About Section -->
	<div class="w3-container" style="padding:128px 10%">
		<div class="w3-container" >
			<div class="w3-container w3-center" style="height:350px">
				<div id="generatedData" style="display:block">
				<% 
				String generateSuccess = (String) request.getAttribute("generateSuccess");
				String generatePathRel = (String) request.getAttribute("generatePathRel");
				String generatePathAbs = (String) request.getAttribute("generatePathAbs");
				if(generateSuccess != null) 
				{
					if(generateSuccess.equals("True"))
					{
				%>
					<img src="<%out.println(generatePathRel);%>" style="width:50%; margin-top:64px" alt="generated"> </br>
					<div>
						<audio controls preload="none" style="margin:16px 0px 0px 32px">
							<source src="<%out.println(generatePathRel);%>" type="audio/wav">
							Audio element was not supported.
						</audio>
					</div>
				<%	}
				}
				else
				{
				%>
					<img src="README_resources/images/title.png" alt="title">
				<%
				}%>
				</div>
				<div class="w3-jumbo" id="loadSpin" style="margin-top:96px; display:none;">
					<i class="fas fa-spinner fa-spin"></i>
				</div>
			</div>
			
			
			
			<div class="w3-container w3-center">
				<div class="w3-xxlarge">Test model & create data.</div>
				<p class="w3-large">Press 'Generate' button for testing model.</p>
				<a href="generate?type=A" class="w3-btn w3-black anchorNoDeco" onclick="showSpin()"> 
					Generate Type A
				</a>
				<a href="generate?type=B" class="w3-btn w3-black anchorNoDeco" onclick="showSpin()"> 
					Generate Type B
				</a>
			</div>
		</div>
	</div>
	
	
	
	
	
	<!-- Detail Section -->
	<div class="w3-container w3-light-grey" style="padding:128px 20%">
		<h3 class="w3-center w3-xxlarge" style="margin-bottom:64px"> Good result</h3>
		
		<div class="w3-container" style="">
			<div class="w3-xlarge"> DCGAN type A - 1260th training model </div> </br>
			<img src="README_resources/images/bestWork/Piano-DCGAN-trialA1-1260.png" style="width:50%; margin-left:32px" alt="note"> </br>
			<div>
				<audio controls preload="none" style="margin:16px 0px 32px 32px">
					<source src="README_resources/audio/bestWork/Piano-DCGAN-trialA1-1260.mp3" type="audio/mpeg">
					Audio element was not supported.
				</audio>
			</div>
		</div>
		
		<div class="w3-container" style="">
			<div class="w3-xlarge"> DCGAN type B - 30th training model </div> </br>
			<img src="README_resources/images/bestWork/Piano-DCGAN-trialB-30.png" style="width:50%; margin-left:32px" alt="note"> </br>
			<div>
				<audio controls preload="none" style="margin:16px 0px 32px 32px">
					<source src="README_resources/audio/bestWork/Piano-DCGAN-trialB-30.mp3" type="audio/mpeg">
					Audio element was not supported.
				</audio>
			</div>
		</div>
	</div>
		
		
		
		
		
	<div class="w3-container w3-white" style="padding:128px 20%">
		<h3 class="w3-center w3-xxlarge" style="margin-bottom:64px"> Another selected result</h3>
		
		<div class="w3-container" style="">
			<div class="w3-xlarge"> DCGAN type A - 700th training model </div> </br>
			<img src="README_resources/images/bestWork/Piano-DCGAN-trialA1-700.png" style="width:50%; margin-left:32px" alt="note"> </br>
			<div>
				<audio controls preload="none" style="margin:16px 0px 32px 32px">
					<source src="README_resources/audio/bestWork/Piano-DCGAN-trialA1-700.mp3" type="audio/mpeg">
					Audio element was not supported.
				</audio>
			</div>
		</div>
		
		<div class="w3-container" style="">
			<div class="w3-xlarge"> DCGAN type A - 80th training model </div> </br>
			<img src="README_resources/images/bestWork/Piano-DCGAN-trialA2-80.png" style="width:50%; margin-left:32px" alt="note"> </br>
			<div>
				<audio controls preload="none" style="margin:16px 0px 32px 32px">
					<source src="README_resources/audio/bestWork/Piano-DCGAN-trialA2-80.mp3" type="audio/mpeg">
					Audio element was not supported.
				</audio>
			</div>
		</div>
		
		<div class="w3-container" style="">
			<div class="w3-xlarge"> DCGAN type B - 45th training model </div> </br>
			<img src="README_resources/images/bestWork/Piano-DCGAN-trialB-45.png" style="width:50%; margin-left:32px" alt="note"> </br>
			<div>
				<audio controls preload="none" style="margin:16px 0px 32px 32px">
					<source src="README_resources/audio/bestWork/Piano-DCGAN-trialB-45.mp3" type="audio/mpeg">
					Audio element was not supported.
				</audio>
			</div>
		</div>
		
		<div class="w3-container" style="">
			<div class="w3-xlarge"> DCGAN type B - 60th training model</div> </br>
			<img src="README_resources/images/bestWork/Piano-DCGAN-trialB-60.png" style="width:50%; margin-left:32px" alt="note"> </br>
			<div>
				<audio controls preload="none" style="margin:16px 0px 32px 32px">
					<source src="README_resources/audio/bestWork/Piano-DCGAN-trialB-60.mp3" type="audio/mpeg">
					Audio element was not supported.
				</audio>
			</div>
		</div>
		
		<div class="w3-container" style="">
			<div class="w3-xlarge"> DCGAN type B - 65th training model </div> </br>
			<img src="README_resources/images/bestWork/Piano-DCGAN-trialB-65.png" style="width:50%; margin-left:32px" alt="note"> </br>
			<div>
				<audio controls preload="none" style="margin:16px 0px 32px 32px">
					<source src="README_resources/audio/bestWork/Piano-DCGAN-trialB-65.mp3" type="audio/mpeg">
					Audio element was not supported.
				</audio>
			</div>
		</div>
		
		<div class="w3-container" style="">
			<div class="w3-xlarge"> BEDCGAN - 50th training model </div> </br>
			<img src="README_resources/images/bestWork/Piano-BEDCGAN.png" style="width:15%; margin-left:32px" alt="note"> </br>
			<div>
				<audio controls preload="none" style="margin:16px 0px 32px 32px">
					<source src="README_resources/audio/bestWork/Piano-BEDCGAN.mp3" type="audio/mpeg">
					Audio element was not supported.
				</audio>
			</div>
		</div>
		
		<div class="w3-container" style="">
			<div class="w3-xlarge"> DCGAN with another generator - Mini Batch </div> </br>
			<img src="README_resources/images/bestWork/Piano-MiniBatch.png" style="width:50%; margin-left:32px" alt="note"> </br>
			<div>
				<audio controls preload="none" style="margin:16px 0px 32px 32px">
					<source src="README_resources/audio/bestWork/Piano-MiniBatch.mp3" type="audio/mpeg">
					Audio element was not supported.
				</audio>
			</div>
		</div>
	</div>
	
	
	
	
	
	<!-- Reference -->
	<div class="w3-container w3-black" style="padding:64px 10%">
		<h3>Special thanks to</h3>
		
		<div class="w3-large w3-black" style="margin-top:32px">
			Beck sun woo for translation.</br>
		</div>
	</div>
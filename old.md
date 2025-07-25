<h1 align="center" id="title">HoloDisplay</h1>

debug speechapi:
python3 server/whisper_streaming/transcribe.py --test-harvard --harvard-path /root/holopcscreen/server/whisper_streaming/harvard.wav --debug --model tiny.en

npm i 
pip install librosa soundfile 
pip install faster-whisper websockets



sh fullrush.sh

view screens;
screen -ls 


angular:
screen -r fe

whisper python:
screen -r whisper

express server:
screen -r js


angularjs/ render images/gifs/streams into polygon 
npm start 
4200

webapp : accepts audio trough web
server/index.html
3333 or 4200


python backend/ load transcribe model and pass it 
python3 server/whisper_streaming/optimized_streaming_simple.py --model tiny.en
8765

js server: manage stream and passing transptions/translations
node server/server.js
3000









Future upgrades:
✅ 1. MJPEG Stream over <img> or <video> tag
Essentially an endless sequence of JPEGs over HTTP (multipart/x-mixed-replace)

Compatible with standard <img> tag

Very low latency (better than GIFs)

Easy to integrate — appears like a GIF

html
Copy
Edit
<img src="http://yourserver/mjpeg-stream" />
✅ Works in AngularJS with basic <img> binding
⚠️ No audio, but perfect for visual-only apps

✅ 2. Auto-updating JPG or PNG with cache-busting
Backend updates screen.jpg every second

Frontend reloads every second or faster

html
Copy
Edit
<img ng-src="/screen.jpg?ts={{timestamp}}" />
✅ Very easy to implement
⚠️ Still limited to snapshot feel, but no GIF overhead

✅ 3. Chained MP4 Segments (like HLS Lite)
If MP4 is allowed, you can:

Generate short MP4 clips (1-2s)

Serve them sequentially

Angular plays next when one ends

html
Copy
Edit
<video autoplay loop muted>
  <source ng-src="{{currentSegment}}" type="video/mp4">
</video>
✅ Better quality & compression than GIF
⚠️ Slight delay between segments unless preloaded

✅ 4. Sprite Sheets + Canvas Playback
Combine frames into one large image (like a GIF sprite)

Use JavaScript to animate it on <canvas>

✅ Great compression and full control
⚠️ Requires some JS animation code but works offline

✅ 5. APNG (Animated PNG)
Supports transparency and better quality than GIF

Not supported everywhere but better on Android than you might expect

Could be swapped in place of GIF if allowed

✅ Transparent, high-quality animation
⚠️ File size is large if not optimized





tests
python3 server/fulltest.py







Build your Angular app:
bashng build --prod

Copy the built files to your server directory:
bash
cp -r dist/* ../server/dist/

Start the server:
bashcd ../holodisplay-server
npm start


Old: Speech → Server → Text-to-Image → GIF → Blob → Display
New: Speech → Server → Direct Text Polling → Canvas Rendering







python3 whisper_online.py harvard.wav --model tiny.en --language en --min-chunk-size 1 > out.txt

# Force CPU-only mode and test
CUDA_VISIBLE_DEVICES="" python3 whisper_online.py harvard.wav --model tiny.en --language en --min-chunk-size 1

rm -rf ~/.cache/whisper
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/faster-whisper


# Set memory limits before running
ulimit -v 4194304  # 4GB virtual memory limit
export OMP_NUM_THREADS=1  # Limit OpenMP threads



# Test with smallest model and safe settings
python3 whisper_online.py harvard.wav \
    --model tiny.en \
    --backend faster-whisper \
    --language en \
    --min-chunk-size 2 \
    --offline



    pip install faster-whisper requests soundfile
python3 simpleaudio.py --model tiny.en



Size	Parameters	English-only model	Multilingual model	Required VRAM	Relative speed
tiny	39 M	tiny.en	tiny	~1 GB	~10x
base	74 M	base.en	base	~1 GB	~7x
small	244 M	small.en	small	~2 GB	~4x
medium	769 M	medium.en	medium	~5 GB	~2x
large	1550 M	N/A	large	~10 GB	1x
turbo	809 M	N/A	turbo	~6 GB	~8x


small.zh
medium.en
medium.cn



Whisper backend.

Several alternative backends are integrated. The most recommended one is faster-whisper with GPU support. Follow their instructions for NVIDIA libraries -- we succeeded with CUDNN 8.5.0 and CUDA 11.7. Install with pip install faster-whisper.

Alternative, less restrictive, but slower backend is whisper-timestamped: 

pip install git+https://github.com/linto-ai/whisper-timestamped


Thirdly, it's also possible to run this software from the OpenAI Whisper API. This solution is fast and requires no GPU, just a small VM will suffice, but you will need to pay OpenAI for api access. Also note that, since each audio fragment is processed multiple times, the price will be higher than obvious from the pricing page, so keep an eye on costs while using. Setting a higher chunk-size will reduce costs significantly. Install with: pip install openai , requires Python >=3.8. For running with the openai-api backend, make sure that your OpenAI api key is set in the OPENAI_API_KEY environment variable. For example, before running, do: export OPENAI_API_KEY=sk-xxx with sk-xxx replaced with your api key.



<p align="center"><img src="https://socialify.git.ci/kgabriel-dev/HoloDisplay/image?description=1&amp;font=Raleway&amp;forks=1&amp;issues=1&amp;name=1&amp;owner=1&amp;pattern=Solid&amp;pulls=1&amp;stargazers=1&amp;theme=Auto" alt="project-image"></p>

<p id="description">HoloDisplay is an open-source application that allows you to create the illusion of a hologram on any device using the Pepper's Ghost Effect. With HoloDisplay you can upload your own images and the app will create a structure on your screen that reflects the image to create the illusion of a hologram floating in the air. The app runs in the browser and is compatible with any device that has a screen and you can customize the size and shape of the structure to fit your needs.</p>

<strong>This projects creates images on your display that need to be reflected, preferrably on a translucent material. How you can build this mirror is described in the section: 🧰 Building the mirror</strong>

<h2>Project Screenshots:</h2>

<img src="https://i.ibb.co/dQJnFbP/displayed-images.png" alt="project-screenshot" width="590">

  
  
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   Easy to use: HoloDisplay is user-friendly. You can upload your images and the app takes care of the rest.
*   Responsive Design: The app is designed to be responsive and compatible with any device that has a screen. That means you can create holograms on your smartphone tablet or desktop computer.
*   Open-Source: Holo Display is an open-source project meaning that anyone can contribute to the development of the app. This also means that the app is free to use.
*   Cross-Browser-Compatibility: You can use the app in every browser.


<h2>🛠️ How to use the software</h2>
You can find a working version of this project on my website <a href="https://hologram.kgabriel.dev/en-US/">https://hologram.kgabriel.dev/en-US/</a>
<br /><br />
Another option is to use the executable file found in the folder `executables`. This file starts a simple webserver and opens the correct website (<a href="http://localhost:5000/en-US/">localhost:5000/en-US/</a>).
<br /><br />
You can also download this source code and build the software yourself. Follow these steps to do so:
<ol>
<li>Download the source code / clone the repository.</li>
<li>Download and install NodeJS to use its package manager NPM. You can find the latest version on <a href="https://nodejs.org/en/download">nodejs.org/en/download</a></li>
<li>Navigate into the root folder of this project and open a terminal there.</li>
<li>Run the command "npm install" to install all required packages.</li>
<li>In the angular.json file in the root folder, change the option "localize" under "projects -> HoloDisplay -> architect -> build -> options" to your desired language. Currently available are English (enter this: ["en-US"]) and German (enter this: ["de-DE"]).</li>
<li>In your terminal, run the command "ng serve". After the command finishes, the project is available as a website under <a href="http://localhost:4200">localhost:4200</a>.</li>
</ol>
  

<h2>🧰 Building the mirror</h2>
To make use of this project, you need a mirror that reflects the images created by the app. You can build this mirror yourself. Here's how you can build a mirror for the standard display method (called "Standard Method" in the app):
<ol>
<li>Get a transparent material. This can be a transparent plastic sheet or a glass pane. The material should be as transparent as possible, but also reflect some of the light.</li>
<li>Configure the settings so they match your needs. Important are the settings "Size of the polygon in the middle" and "Number of sides / images".</li>
<li>Click on the calculator icon in the bottom right of the screen. This will open a popup where you have to enter some more settings.</li>
<li>Click on the green button with the text "Generate image". This will download an image showing you the form the tiles of your mirror need to have.</li>
<li>Cut this form out of the material you chose for your mirror. You need as many tiles as you have set the number of sides/images.</li>
<li>Lay the tiles edge to edge, so they start forming a circle. The connect the edges using some tape. Then connect the outer two edges, so your mirror forms.</li>
</ol>

Here is an image of a mirror with 4 sides:<br />
<img src="https://upload.wikimedia.org/wikipedia/commons/e/e2/Pyramid_holographic_3D_holographic_projection_phone_projector_3D_holographic_projection_3D_mobile_phone_naked_eye_3D_pyramid.jpg" alt="image of a mirror" width="300px"><br />&copy; Image licensed by Karthick98 under the <a href="https://creativecommons.org/licenses/by-sa/4.0/deed.en">Creative Commons Attribution-Share Alike 4.0 International license</a>, uploaded on wikipedia.org.

  
<h2>💻 Built with</h2>

Technologies used in the project:

*   Angular 15
*   TailwindCSS

<h2>🛡️ License:</h2>

This project is licensed under the MIT License, you can find the full license text in the file <a href="https://github.com/kgabriel-dev/HoloDisplay/blob/master/LICENSE">LICENSE</a>.

import React, { useState, useRef, useReducer } from 'react';
import './App.css';
import FadeIn from "react-fade-in";
import Lottie from "react-lottie";
import * as legoData from "./legoloading.json";


const stateMachine = {
 initial: "initial",
 states: {
 initial: { on: { next: "imageReady", text: 'Upload Image' } },
 imageReady: { on: { next: "identifying" }, text: 'Predict Caption', showImage: true },
 }
};

const defaultOptions = {
    loop: true,
    autoplay: true,
    animationData: legoData.default,
    rendererSettings: {
    preserveAspectRatio: "xMidYMid slice"
    }
};

const reducer = (currentState, event) => stateMachine.states[currentState].on[event] || stateMachine.initial;

const App = () => {
  const [imageUrl, setImageUrl] = useState(null);
  const [file,setFiles] = useState();
  const [results, setResults] = useState();
  const [prediction, setPrediction] = useState();
  const inputRef = useRef();
  const imageRef = useRef();

  const state = {
    selectedFile:null
  }

  const [appState, dispatch] = useReducer(reducer, stateMachine.initial);

  const next = () => dispatch("next")


  const [model, setModel] = useState(null)

  const handleUpload = event => {
                                 const { files } = event.target;
                                 if (files.length > 0) {
                                 const url = URL.createObjectURL(files[0])
                                 setImageUrl(url)
                                 setFiles(files[0])
                                 next();
                                 }
                                }

  const upload_to_server = async (e) => {
                                  e.preventDefault();
                                  console.log("Sending request");
                                  setResults("true");
                                  const fd = new FormData();
                                  fd.append('file', file);
                                  fetch("https://image-captioning-hyakkbd4gq-uw.a.run.app/predict",{
                                  method: 'POST',
                                  body: fd}).then(res => res.json())
                                 .then(json => {
                                  setPrediction(json);console.log(json);
                                 })
                                 .catch(err => console.error(err));
                               }


  const buttonProps = {
                       initial: { text: "Upload Image", action: () => inputRef.current.click() },
                       imageReady: { text: "Predict Caption", action: (e) => upload_to_server(e) },

                     };

  const { showImage = false } = stateMachine.states[appState]

  return (
    <div>
    <h1>
    	Image Captioning: Making AI See and Understand
    </h1>
    <p>Image Captioning is the process of generating textual description of an image. It is one of the coolest task in Machine Learning which utilizes both Natural Language Processing and Computer Vision to generate the captions. If you think about it, we're actually teaching AI to learn to see and understand... one more step closer to ultron? (JK)</p>
    <h2>
      Upload an image and click predict caption to see how AI would describe your image
    </h2>
    { showImage &&
   <img
   src={imageUrl}
   alt="upload-preview"
   ref={imageRef}
   />
  }
  { !showImage &&
  <button onClick={buttonProps[appState].action}>{buttonProps[appState].text}</button>
  }
  { showImage && !prediction &&
  <button onClick={buttonProps[appState].action}>{buttonProps[appState].text}</button>
  }
  { results && !prediction &&
    <FadeIn>
      <div class="d-flex justify-content-center align-items-center">
        <h1>The AI bot is still naive, give it around 30-40 seconds to scan and understand the image</h1>
        <Lottie options={defaultOptions} height={120} width={120} />
      </div>
    </FadeIn>
  }
  <input type="file" accept="image/*" capture="camera" onChange={handleUpload} ref={inputRef}></input>
  {
    prediction &&
    <h1>{prediction}</h1>
  }
</div>)
}
export default App;

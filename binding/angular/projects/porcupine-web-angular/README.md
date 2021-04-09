# porcupine-web-angular

Angular service for Porcupine for Web.

## Porcupine

Porcupine is a highly accurate and lightweight wake word engine. It enables building always-listening voice-enabled
applications.

> Porcupine wake word models for all major voice assistants (`Alexa`, `Hey Google`, `Ok Google`, and `Hey Siri`) are
> available for free (under Apache 2.0)

## Introduction

The Porcupine SDK for Angular is based on the Porcupine SDK for Web. The library provides an Angular service called `PorcupineService`. The package will take care of microphone access and audio downsampling (via `@picovoice/web-voice-processor`) and provide a wake word detection event to which your application can subscribe.

## Compatibility

The Picovoice SDKs for Web are powered by WebAssembly (WASM), the Web Audio API, and Web Workers.

All modern browsers (Chrome/Edge/Opera, Firefox, Safari) are supported, including on mobile. Internet Explorer is _not_ supported.

Using the Web Audio API requires a secure context (HTTPS connection), with the exception of `localhost`, for local development.

## Installation

Use `npm` or `yarn` to install the package and its peer dependencies. Each spoken language (e.g. 'en', 'de') is a separate package. For this example we'll use English:

`yarn add @picovoice/porcupine-web-angular @picovoice/web-voice-processor @picovoice/porcupine-web-en-worker`

(or)

`npm install @picovoice/porcupine-web-angular @picovoice/web-voice-processor @picovoice/porcupine-web-en-worker`

## Usage

In your Angular component, add the PorcupineService. The PorcupineService has a detection event to which you can subscribe:

```typescript
import { Subscription } from "rxjs"
import { PorcupineService } from "@picovoice/porcupine-web-angular"

...

  constructor(private porcupineService: PorcupineService) {
    // Subscribe to Porcupine Keyword detections
    // Store each detection so we can display it in an HTML list
    this.porcupineDetection = porcupineService.detection$.subscribe(
      keywordLabel => console.log(`Porcupine Detected "${keywordLabel}"`))
  }
```

We need to initialize Porcupine to tell it which keywords we want to listen to (and at what sensitivity). We can use the Angular lifecycle hooks `ngOnInit` and `ngOnDestroy` to start up and later tear down the Porcupine engine.

**Important Note** The @picovoice/porcupine-web-${LANGUAGE}-\* series of packages are on the order of ~1-2MB, as they contain the entire Voice AI model. Typically, you do _not_ want to import these statically, as your application bundle will be much larger than recommended. Instead, use dynamic imports so that the chunk is lazy-loaded:

```typescript
  async ngOnInit() {
    // Load Porcupine worker chunk with specific language model (large ~1-2MB chunk; dynamically imported)
    const porcupineFactoryEn = (await import('@picovoice/porcupine-web-en-worker')).PorcupineWorkerFactory
    // Initialize Porcupine Service
    try {
      await this.porcupineService.init(porcupineFactoryEn,
      {porcupineFactoryArgs: [{ builtin: "Okay Google", sensitivity: 0.65 }, { builtin: "Picovoice" }]})
      console.log("Porcupine is now loaded and listening")
    }
    catch (error) {
      console.error(error)
    }
  }

  ngOnDestroy() {
    this.porcupineDetection.unsubscribe()
    this.porcupineService.release()
  }

```

Upon mounting, the component will request microphone permission from the user, instantiate the audio stream, start up an instance of Porcupine, and listen for both "Picovoice" and "Okay Google". When the words are detected, they will be logged to the console.

The "Okay Google" word is listening at a sensitivity of 0.65, whereas "Picovoice" is using the default (0.5). Sensitivity is a value in the range [0,1] which controls the tradeoff between miss rate and false alarm.

### Custom wake words

Each language includes a set of built-in keywords. The quickest way to get started is to use one of those. The builtin keywords are licensed under Apache-2.0 and are completely free to use.

Custom wake words are generated using [Picovoice Console](https://picovoice.ai/console/). They are trained from text using transfer learning into bespoke Porcupine keyword files with a `.ppn` extension. The target platform is WebAssembly (WASM), as that is what backs the Angular library.

Convert the `.ppn` file to base64 and provide it as an argument to Porcupine as below. You will need to also provide a label so that the PorcupineService can tell you which keyword occurred ("Deep Sky Blue", in this case):

```typescript
const DEEP_SKY_BLUE_PPN_64 = "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"


...
  // Listen for "Deep Sky Blue": pass in a base64-encoded string of the .ppn file:
  await this.porcupineService.init(porcupineFactoryEn,
  {porcupineFactoryArgs: [{ custom: "Deep Sky Blue", base64: DEEP_SKY_BLUE_PPN_64 }]})

...
```

You may wish to store the base64 string in a separate JavaScript file and export it to keep your application code separate.

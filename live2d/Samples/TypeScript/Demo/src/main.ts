/**
 * Copyright(c) Live2D Inc. All rights reserved.
 *
 * Use of this source code is governed by the Live2D Open Software license
 * that can be found at https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html.
 */

import { LAppDelegate } from './lappdelegate';
import * as LAppDefine from './lappdefine';
import { io } from 'socket.io-client';
import { LAppLive2DManager } from './lapplive2dmanager';
import { wsURL } from '../../../../../frontend/src/Settings/Constants'

const socket = io(wsURL, {
  transports: ["websocket"],
});

socket.on("live2dchangemodelmale", (session_token) => {
  console.log("Try to switch to male", session_token, LAppDelegate.getInstance().getSessionToken())
  if (session_token !== LAppDelegate.getInstance().getSessionToken()) {
    return;
  }
  console.log("Switch to male")
  LAppDelegate.getInstance().setRenderView(false);
  LAppLive2DManager.getInstance().changeScene(0);
  LAppDelegate.getInstance().setRenderView(true);
});

socket.on("live2dchangemodelfemale", (session_token) => {
  console.log("Try to switch to female", session_token, LAppDelegate.getInstance().getSessionToken())
  if (session_token !== LAppDelegate.getInstance().getSessionToken()) {
    return;
  }
  console.log("Switch to female")
  LAppDelegate.getInstance().setRenderView(false);
  LAppLive2DManager.getInstance().changeScene(1);
  LAppDelegate.getInstance().setRenderView(true);
});

socket.on("live2dlipsync", (session_token) => {
  if (session_token !== LAppDelegate.getInstance().getSessionToken()) {
    return;
  }
  console.log("Lipsync start")
  LAppLive2DManager.getInstance().getModel(0)._wavFileHandler.start(
    LAppDefine.ResourcesPath + "output_sounds/" + session_token + "/output_sound.wav"
  );
});

/**
 * ブラウザロード後の処理
 */
window.onload = (): void => {
  // create the application instance
  if (LAppDelegate.getInstance().initialize() == false) {
    return;
  }

  LAppDelegate.getInstance().run();

  const parentURL = new URL(document.location.href)
  LAppDelegate.getInstance().setSessionToken(parentURL.searchParams.get("sessiontoken"));

  console.log(`The value of the "session_token" parameter is "${LAppDelegate.getInstance().getSessionToken()}"`);
  socket.emit("resendgenderinfo", LAppDelegate.getInstance().getSessionToken());
};

/**
 * 終了時の処理
 */
window.onbeforeunload = (): void => LAppDelegate.releaseInstance();

/**
 * Process when changing screen size.
 */
window.onresize = () => {
  if (LAppDefine.CanvasSize === 'auto') {
    LAppDelegate.getInstance().onResize();
  }
};

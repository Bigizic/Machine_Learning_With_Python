"use strict";(self.webpackChunk_JUPYTERLAB_CORE_OUTPUT=self.webpackChunk_JUPYTERLAB_CORE_OUTPUT||[]).push([[6433],{36433:(e,s,t)=>{t.r(s),t.d(s,{BaseKernel:()=>r,FALLBACK_KERNEL:()=>f,IKernelSpecs:()=>w,IKernels:()=>y,KernelSpecs:()=>k,Kernels:()=>m});var n=t(50558),i=t(2159);class r{constructor(e){this._history=[],this._executionCount=0,this._isDisposed=!1,this._disposed=new i.Signal(this),this._parentHeader=void 0,this._parent=void 0;const{id:s,name:t,location:n,sendMessage:r}=e;this._id=s,this._name=t,this._location=n,this._sendMessage=r}get ready(){return Promise.resolve()}get isDisposed(){return this._isDisposed}get disposed(){return this._disposed}get id(){return this._id}get name(){return this._name}get location(){return this._location}get executionCount(){return this._executionCount}get parentHeader(){return this._parentHeader}get parent(){return this._parent}dispose(){this.isDisposed||(this._isDisposed=!0,this._disposed.emit(void 0))}async handleMessage(e){switch(this._busy(e),this._parent=e,e.header.msg_type){case"kernel_info_request":await this._kernelInfo(e);break;case"execute_request":await this._execute(e);break;case"input_reply":this.inputReply(e.content);break;case"inspect_request":await this._inspect(e);break;case"is_complete_request":await this._isCompleteRequest(e);break;case"complete_request":await this._complete(e);break;case"history_request":await this._historyRequest(e);break;case"comm_open":await this.commOpen(e);break;case"comm_msg":await this.commMsg(e);break;case"comm_close":await this.commClose(e)}this._idle(e)}stream(e,s=void 0){var t;const i=void 0!==s?s:this._parentHeader,r=n.KernelMessage.createMessage({channel:"iopub",msgType:"stream",session:null!==(t=null==i?void 0:i.session)&&void 0!==t?t:"",parentHeader:i,content:e});this._sendMessage(r)}displayData(e,s=void 0){var t,i;const r=void 0!==s?s:this._parentHeader;e.metadata=null!==(t=e.metadata)&&void 0!==t?t:{};const a=n.KernelMessage.createMessage({channel:"iopub",msgType:"display_data",session:null!==(i=null==r?void 0:r.session)&&void 0!==i?i:"",parentHeader:r,content:e});this._sendMessage(a)}inputRequest(e,s=void 0){var t;const i=void 0!==s?s:this._parentHeader,r=n.KernelMessage.createMessage({channel:"stdin",msgType:"input_request",session:null!==(t=null==i?void 0:i.session)&&void 0!==t?t:"",parentHeader:i,content:e});this._sendMessage(r)}publishExecuteResult(e,s=void 0){var t;const i=void 0!==s?s:this._parentHeader,r=n.KernelMessage.createMessage({channel:"iopub",msgType:"execute_result",session:null!==(t=null==i?void 0:i.session)&&void 0!==t?t:"",parentHeader:i,content:e});this._sendMessage(r)}publishExecuteError(e,s=void 0){var t;const i=void 0!==s?s:this._parentHeader,r=n.KernelMessage.createMessage({channel:"iopub",msgType:"error",session:null!==(t=null==i?void 0:i.session)&&void 0!==t?t:"",parentHeader:i,content:e});this._sendMessage(r)}updateDisplayData(e,s=void 0){var t;const i=void 0!==s?s:this._parentHeader,r=n.KernelMessage.createMessage({channel:"iopub",msgType:"update_display_data",session:null!==(t=null==i?void 0:i.session)&&void 0!==t?t:"",parentHeader:i,content:e});this._sendMessage(r)}clearOutput(e,s=void 0){var t;const i=void 0!==s?s:this._parentHeader,r=n.KernelMessage.createMessage({channel:"iopub",msgType:"clear_output",session:null!==(t=null==i?void 0:i.session)&&void 0!==t?t:"",parentHeader:i,content:e});this._sendMessage(r)}handleComm(e,s,t,i,r=void 0){var a;const o=void 0!==r?r:this._parentHeader,c=n.KernelMessage.createMessage({channel:"iopub",msgType:e,session:null!==(a=null==o?void 0:o.session)&&void 0!==a?a:"",parentHeader:o,content:s,metadata:t,buffers:i});this._sendMessage(c)}_idle(e){const s=n.KernelMessage.createMessage({msgType:"status",session:e.header.session,parentHeader:e.header,channel:"iopub",content:{execution_state:"idle"}});this._sendMessage(s)}_busy(e){const s=n.KernelMessage.createMessage({msgType:"status",session:e.header.session,parentHeader:e.header,channel:"iopub",content:{execution_state:"busy"}});this._sendMessage(s)}async _kernelInfo(e){const s=await this.kernelInfoRequest(),t=n.KernelMessage.createMessage({msgType:"kernel_info_reply",channel:"shell",session:e.header.session,parentHeader:e.header,content:s});this._sendMessage(t)}async _historyRequest(e){const s=e,t=n.KernelMessage.createMessage({msgType:"history_reply",channel:"shell",parentHeader:s.header,session:e.header.session,content:{status:"ok",history:this._history}});this._sendMessage(t)}_executeInput(e){const s=e,t=s.content.code,i=n.KernelMessage.createMessage({msgType:"execute_input",parentHeader:s.header,channel:"iopub",session:e.header.session,content:{code:t,execution_count:this._executionCount}});this._sendMessage(i)}async _execute(e){const s=e,t=s.content;t.store_history&&this._executionCount++,this._parentHeader=s.header,this._executeInput(s),t.store_history&&this._history.push([0,0,t.code]);const i=await this.executeRequest(s.content),r=n.KernelMessage.createMessage({msgType:"execute_reply",channel:"shell",parentHeader:s.header,session:e.header.session,content:i});this._sendMessage(r)}async _complete(e){const s=e,t=await this.completeRequest(s.content),i=n.KernelMessage.createMessage({msgType:"complete_reply",parentHeader:s.header,channel:"shell",session:e.header.session,content:t});this._sendMessage(i)}async _inspect(e){const s=e,t=await this.inspectRequest(s.content),i=n.KernelMessage.createMessage({msgType:"inspect_reply",parentHeader:s.header,channel:"shell",session:e.header.session,content:t});this._sendMessage(i)}async _isCompleteRequest(e){const s=e,t=await this.isCompleteRequest(s.content),i=n.KernelMessage.createMessage({msgType:"is_complete_reply",parentHeader:s.header,channel:"shell",session:e.header.session,content:t});this._sendMessage(i)}}var a=t(6128),o=t(81259),c=t(70552),l=t(21961),h=t(44700);new Error("timeout while waiting for mutex to become available"),new Error("mutex already locked");const u=new Error("request for lock canceled");var d=function(e,s,t,n){return new(t||(t=Promise))((function(i,r){function a(e){try{c(n.next(e))}catch(e){r(e)}}function o(e){try{c(n.throw(e))}catch(e){r(e)}}function c(e){var s;e.done?i(e.value):(s=e.value,s instanceof t?s:new t((function(e){e(s)}))).then(a,o)}c((n=n.apply(e,s||[])).next())}))};class _{constructor(e,s=u){if(this._maxConcurrency=e,this._cancelError=s,this._queue=[],this._waiters=[],e<=0)throw new Error("semaphore must be initialized to a positive value");this._value=e}acquire(){const e=this.isLocked(),s=new Promise(((e,s)=>this._queue.push({resolve:e,reject:s})));return e||this._dispatch(),s}runExclusive(e){return d(this,void 0,void 0,(function*(){const[s,t]=yield this.acquire();try{return yield e(s)}finally{t()}}))}waitForUnlock(){return d(this,void 0,void 0,(function*(){return this.isLocked()?new Promise((e=>this._waiters.push({resolve:e}))):Promise.resolve()}))}isLocked(){return this._value<=0}release(){if(this._maxConcurrency>1)throw new Error("this method is unavailable on semaphores with concurrency > 1; use the scoped release returned by acquire instead");if(this._currentReleaser){const e=this._currentReleaser;this._currentReleaser=void 0,e()}}cancel(){this._queue.forEach((e=>e.reject(this._cancelError))),this._queue=[]}_dispatch(){const e=this._queue.shift();if(!e)return;let s=!1;this._currentReleaser=()=>{s||(s=!0,this._value++,this._resolveWaiters(),this._dispatch())},e.resolve([this._value--,this._currentReleaser])}_resolveWaiters(){this._waiters.forEach((e=>e.resolve())),this._waiters=[]}}class p{constructor(e){this._semaphore=new _(1,e)}acquire(){return e=this,s=void 0,n=function*(){const[,e]=yield this._semaphore.acquire();return e},new((t=void 0)||(t=Promise))((function(i,r){function a(e){try{c(n.next(e))}catch(e){r(e)}}function o(e){try{c(n.throw(e))}catch(e){r(e)}}function c(e){var s;e.done?i(e.value):(s=e.value,s instanceof t?s:new t((function(e){e(s)}))).then(a,o)}c((n=n.apply(e,s||[])).next())}));var e,s,t,n}runExclusive(e){return this._semaphore.runExclusive((()=>e()))}isLocked(){return this._semaphore.isLocked()}waitForUnlock(){return this._semaphore.waitForUnlock()}release(){this._semaphore.release()}cancel(){return this._semaphore.cancel()}}var g=t(64145);const v=c.supportedKernelWebSocketProtocols.v1KernelWebsocketJupyterOrg;class m{constructor(e){this._kernels=new a.ObservableMap,this._clients=new a.ObservableMap,this._kernelClients=new a.ObservableMap;const{kernelspecs:s}=e;this._kernelspecs=s}async startNew(e){const{id:s,name:t,location:n}=e,i=this._kernelspecs.factories.get(t);if(!i)return{id:s,name:t};const r=new p,a=(e,s,t)=>{var n;const i=this._kernels.get(e);if(!i)throw Error(`No kernel ${e}`);this._clients.set(s,t),null===(n=this._kernelClients.get(e))||void 0===n||n.add(s),t.on("message",(async e=>{let s;if(e instanceof ArrayBuffer)e=new Uint8Array(e).buffer,s=(0,o.deserialize)(e,v);else{if("string"!=typeof e)return;{const t=(new TextEncoder).encode(e);s=(0,o.deserialize)(t.buffer,v)}}"input_reply"===s.header.msg_type?i.handleMessage(s):(async e=>{await r.runExclusive((async()=>{await i.ready,await i.handleMessage(e)}))})(s)}));const a=()=>{var t;this._clients.delete(s),null===(t=this._kernelClients.get(e))||void 0===t||t.delete(s)};i.disposed.connect(a),t.onclose=a},c=null!=s?s:l.UUID.uuid4(),u=`${m.WS_BASE_URL}api/kernels/${c}/channels`,d=this._kernels.get(c);if(d)return{id:d.id,name:d.name};const _=await i({id:c,sendMessage:e=>{const s=e.header.session,t=this._clients.get(s);if(!t)return void console.warn(`Trying to send message on removed socket for kernel ${c}`);const n=(0,o.serialize)(e,v);if("iopub"!==e.channel)t.send(n);else{const e=this._kernelClients.get(c);null==e||e.forEach((e=>{var s;null===(s=this._clients.get(e))||void 0===s||s.send(n)}))}},name:t,location:n});this._kernels.set(c,_),this._kernelClients.set(c,new Set);const g=new h.Server(u,{mock:!1,selectProtocol:()=>v});return g.on("connection",(e=>{var s;const t=null!==(s=new URL(e.url).searchParams.get("session_id"))&&void 0!==s?s:"";a(c,t,e)})),g.on("close",(()=>{this._clients.keys().forEach((e=>{var s;const t=this._clients.get(e);(null==t?void 0:t.readyState)===WebSocket.CLOSED&&(this._clients.delete(e),null===(s=this._kernelClients.get(c))||void 0===s||s.delete(e))}))})),_.disposed.connect((()=>{g.close(),this._kernels.delete(c),this._kernelClients.delete(c)})),{id:_.id,name:_.name}}async restart(e){const s=this._kernels.get(e);if(!s)throw Error(`Kernel ${e} does not exist`);const{id:t,name:n,location:i}=s;return s.dispose(),this.startNew({id:t,name:n,location:i})}async list(){return[...this._kernels.values()].map((e=>({id:e.id,name:e.name})))}async shutdown(e){var s;null===(s=this._kernels.delete(e))||void 0===s||s.dispose()}async get(e){return this._kernels.get(e)}}!function(e){e.WS_BASE_URL=g.PageConfig.getBaseUrl().replace(/^http/,"ws")}(m||(m={}));const y=new l.Token("@jupyterlite/kernel:IKernels"),f="javascript",w=new l.Token("@jupyterlite/kernel:IKernelSpecs");class k{constructor(){this._specs=new Map,this._factories=new Map}get specs(){return 0===this._specs.size?null:{default:this.defaultKernelName,kernelspecs:Object.fromEntries(this._specs)}}get defaultKernelName(){let e=g.PageConfig.getOption("defaultKernelName");if(!e&&this._specs.size){const s=Array.from(this._specs.keys());s.sort(),e=s[0]}return e||f}get factories(){return this._factories}register(e){const{spec:s,create:t}=e;this._specs.set(s.name,s),this._factories.set(s.name,t)}}}}]);
//# sourceMappingURL=6433.b126e2d.js.map
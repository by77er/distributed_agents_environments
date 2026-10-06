# API reference

Every public name, grouped by module, alphabetically, for anyone who knows what they are looking for. Types
and defaults appear as written in the source. Generated from the source by `scripts/generate_reference.py`;
do not edit by hand.

**Read first:** [Start here](../start/README.md), and the section for your task in the
[documentation home](../README.md), for how the pieces fit together.

## Contents

- **[`rollout.harness`](#rolloutharness)** — Writing tasks, agents and programs; runners; memory; tool sets. [`Agent`](#agent), [`agent_program`](#agent_program), [`AgentProgram`](#agentprogram), [`bind`](#bind), [`Blobs`](#blobs), [`Capacity`](#capacity), [`CompactingAgent`](#compactingagent), [`ContextHints`](#contexthints), [`DeduplicatingToolSet`](#deduplicatingtoolset), [`DirectModel`](#directmodel), [`Effects`](#effects), [`End`](#end), [`Ending`](#ending), [`EndpointModel`](#endpointmodel), [`FileBlobStore`](#fileblobstore), [`History`](#history), [`HistoryShape`](#historyshape), [`instantiate`](#instantiate), [`InvalidObservation`](#invalidobservation), [`Lease`](#lease), [`LeaseRefused`](#leaserefused), [`Leases`](#leases), [`Memory`](#memory), [`MemoryLeases`](#memoryleases), [`Model`](#model), [`ModelBinding`](#modelbinding), [`ModelSample`](#modelsample), [`ModelSlot`](#modelslot), [`Mount`](#mount), [`Network`](#network), [`NoCapacity`](#nocapacity), [`Observation`](#observation), [`Pool`](#pool), [`PoolBinding`](#poolbinding), [`Process`](#process), [`Program`](#program), [`ProgramReference`](#programreference), [`Provider`](#provider), [`Reach`](#reach), [`RecordedEndpoints`](#recordedendpoints), [`RecordedModel`](#recordedmodel), [`register`](#register), [`resolve`](#resolve), [`rollout`](#rollout), [`RunBinding`](#runbinding), [`RunContext`](#runcontext), [`RunHandle`](#runhandle), [`RunHooks`](#runhooks), [`Runner`](#runner), [`RunOutcome`](#runoutcome), [`RunSpecification`](#runspecification), [`RunStatus`](#runstatus), [`SamplingParameters`](#samplingparameters), [`Sandbox`](#sandbox), [`SandboxLimits`](#sandboxlimits), [`SandboxLost`](#sandboxlost), [`SandboxPool`](#sandboxpool), [`SandboxSpec`](#sandboxspec), [`Scratch`](#scratch), [`Task`](#task), [`tool`](#tool), [`ToolBinding`](#toolbinding), [`Tools`](#tools), [`ToolSet`](#toolset), [`Turn`](#turn), [`with_row`](#with_row)
- **[`rollout.contracts`](#rolloutcontracts)** — Types that cross layers: canonical content, identifiers, digests, effects, events. [`address_of`](#address_of), [`AddressableEndpoint`](#addressableendpoint), [`arguments_digest`](#arguments_digest), [`BlobReference`](#blobreference), [`Block`](#block), [`canonical_json`](#canonical_json), [`CapabilityContract`](#capabilitycontract), [`Conflict`](#rolloutcontractsconflict), [`context_digests`](#context_digests), [`ContextDelta`](#contextdelta), [`ContextOverflow`](#contextoverflow), [`ContractModel`](#contractmodel), [`ContractViolation`](#contractviolation), [`digest`](#digest), [`effect_id`](#effect_id), [`EffectIdentity`](#effectidentity), [`EffectKind`](#effectkind), [`EffectStatus`](#effectstatus), [`EMPTY_DIGEST`](#empty_digest), [`FinishReason`](#finishreason), [`FrozenSequence`](#frozensequence), [`InternalError`](#internalerror), [`Media`](#media), [`Message`](#message), [`message_digest`](#message_digest), [`ModelAddress`](#modeladdress), [`ModelEndpoint`](#modelendpoint), [`ModelEndpointError`](#modelendpointerror), [`NamedToolChoice`](#namedtoolchoice), [`new_run_id`](#new_run_id), [`new_ulid`](#new_ulid), [`Overloaded`](#overloaded), [`Reasoning`](#reasoning), [`ReasoningScope`](#reasoningscope), [`ResultBlock`](#resultblock), [`RetryClass`](#retryclass), [`Role`](#role), [`RUN_EVENT_SCHEMA_VERSION`](#run_event_schema_version), [`RunEvent`](#runevent), [`RunEventType`](#runeventtype), [`RunFailureClass`](#runfailureclass), [`SampleLink`](#samplelink), [`SampleRequest`](#samplerequest), [`SampleResult`](#sampleresult), [`session_id`](#session_id), [`SessionIdentity`](#sessionidentity), [`spec_hash`](#spec_hash), [`TERMINAL_EVENT_TYPES`](#terminal_event_types), [`Text`](#text), [`ToolCall`](#toolcall), [`ToolChoice`](#toolchoice), [`ToolChoiceMode`](#toolchoicemode), [`ToolResult`](#toolresult), [`ToolResultBlock`](#toolresultblock), [`ToolSpecification`](#toolspecification), [`Usage`](#usage)
- **[`rollout.environment`](#rolloutenvironment)** — What a run trains on and an eval measures: rows, starts, eval data, what results say. [`binding_for`](#binding_for), [`Description`](#description), [`drawn`](#drawn), [`Environment`](#environment), [`first_program`](#first_program), [`held_out`](#held_out), [`Row`](#row), [`Start`](#start), [`start_key`](#start_key), [`train_start`](#train_start)
- **[`rollout.curriculum`](#rolloutcurriculum)** — Which row to train on next, and gates on evals. [`Curriculum`](#curriculum), [`curriculum_of`](#curriculum_of), [`GroupResult`](#groupresult), [`solved_share`](#solved_share)
- **[`rollout.local`](#rolloutlocal)** — The runner in this process. [`EndpointFactory`](#endpointfactory), [`LocalRunContext`](#localruncontext), [`LocalRunHandle`](#localrunhandle), [`LocalRunner`](#localrunner), [`RewardAssignment`](#rewardassignment)
- **[`rollout.testing`](#rollouttesting)** — Test doubles: a scripted model endpoint and helpers. [`events_of`](#rollouttestingevents_of), [`FakeSandbox`](#fakesandbox), [`FakeSandboxes`](#fakesandboxes), [`LedgerEndpoint`](#ledgerendpoint), [`local_run`](#local_run), [`payload`](#payload), [`ScriptedModelEndpoint`](#scriptedmodelendpoint), [`ScriptedReply`](#scriptedreply), [`tool_call_reply`](#tool_call_reply), [`until`](#until)
- **[`rollout_train.rollouts`](#rollout_trainrollouts)** — Episodes a run asks for in the ledger, claimed and played by runners, and read back. [`Episode`](#episode), [`EpisodeRunner`](#episoderunner), [`episodes_of`](#episodes_of), [`events_of`](#rollout_trainrolloutsevents_of), [`Hooks`](#hooks), [`loaded`](#rollout_trainrolloutsloaded), [`Outcome`](#outcome), [`Plan`](#rollout_trainrolloutsplan), [`plan`](#plan), [`playing`](#playing), [`Record`](#record), [`Recorded`](#recorded), [`stored`](#rollout_trainrolloutsstored), [`Trajectory`](#trajectory)
- **[`rollout_train.sandboxes`](#rollout_trainsandboxes)** — Sandboxes' leases beside the ledger, each ending with its episode's claim. [`admits`](#admits), [`ended`](#ended), [`ending`](#ending), [`FileLeases`](#fileleases), [`keep`](#keep), [`leases_of`](#leases_of), [`pool_scope`](#pool_scope), [`sweep`](#sweep)
- **[`rollout_train`](#rollout_train)** — The training loop, the group algorithm, evals, and what they ask of a trainer. [`Algorithm`](#algorithm), [`algorithm_for`](#algorithm_for), [`Batch`](#batch), [`Budget`](#budget), [`Changeable`](#changeable), [`Checkpoint`](#checkpoint), [`Checkpoints`](#checkpoints), [`Colocated`](#colocated), [`Dataset`](#dataset), [`dataset_of`](#dataset_of), [`Distillations`](#distillations), [`Distilled`](#distilled), [`edit_suite`](#edit_suite), [`evaluate`](#evaluate), [`Fence`](#fence), [`Fenced`](#fenced), [`FileLedger`](#fileledger), [`Files`](#files), [`Follower`](#follower), [`group_advantages`](#group_advantages), [`Grpo`](#grpo), [`Keeps`](#keeps), [`Labelled`](#labelled), [`Ledger`](#ledger), [`Made`](#made), [`make_dataset`](#make_dataset), [`make_suite`](#make_suite), [`Manifest`](#manifest), [`Pair`](#pair), [`Preferences`](#preferences), [`record_serving`](#record_serving), [`Remote`](#remote), [`Resident`](#resident), [`Result`](#result), [`results`](#results), [`Retention`](#retention), [`Schedule`](#schedule), [`Serving`](#serving), [`StateLost`](#statelost), [`Step`](#step), [`StepFailed`](#stepfailed), [`Suite`](#suite), [`suite_entry`](#suite_entry), [`suite_for`](#suite_for), [`suite_of`](#suite_of), [`SuiteEntry`](#suiteentry), [`train`](#train), [`Trained`](#trained), [`trained`](#trained), [`Trainer`](#trainer), [`wanted`](#wanted), [`Weighted`](#weighted)
- **[`rollout_train.inference`](#rollout_traininference)** — Channels: trainable models being served, and what they ask of an engine. [`Channel`](#channel), [`CheckpointServer`](#checkpointserver), [`Connection`](#connection), [`Engine`](#engine), [`Generation`](#generation), [`Limits`](#limits), [`NotLoaded`](#notloaded), [`RemoteChannel`](#remotechannel), [`RemoteEngine`](#remoteengine), [`Route`](#route), [`Routes`](#routes), [`Sampler`](#sampler), [`Scores`](#rollout_traininferencescores), [`Unserved`](#unserved)
- **[`rollout_train.inference.hosts`](#rollout_traininferencehosts)** — Engine hosts: a replica's engines as a Ray actor, serving runs by checkpoint. [`EngineHost`](#enginehost), [`host_spec`](#host_spec), [`HostPausable`](#hostpausable), [`HostServer`](#hostserver), [`HostSpec`](#hostspec), [`started`](#started)
- **[`rollout_train.inference.api`](#rollout_traininferenceapi)** — Channels on hosted APIs: by message, never trained on, spend counted. [`ApiChannel`](#apichannel), [`ATTEMPTS`](#attempts), [`Hosted`](#hosted), [`HostedEndpoint`](#hostedendpoint), [`priced`](#priced)
- **[`rollout_train.recorder`](#rollout_trainrecorder)** — What recording a trainable channel takes: renderers, the thinking budget, segments. [`BEHAVIOUR`](#behaviour), [`ChatTemplateRenderer`](#chattemplaterenderer), [`JsonToolCalls`](#jsontoolcalls), [`rendered`](#rollout_trainrecorderrendered), [`Renderer`](#renderer), [`renders`](#renders), [`sample_turn`](#sample_turn), [`Segment`](#segment), [`segments_of`](#segments_of), [`Span`](#span), [`TeacherScores`](#teacherscores), [`ThinkingFormat`](#thinkingformat), [`TOKEN_LEVEL`](#token_level), [`tokenizer_of`](#tokenizer_of), [`ToolCallFormat`](#toolcallformat), [`XmlFunctionCalls`](#xmlfunctioncalls)
- **[`rollout_train.gateway`](#rollout_traingateway)** — The stateless gateway: samples channels for harnesses and records every turn. [`Attempt`](#attempt), [`ChannelDirectory`](#channeldirectory), [`create_app`](#create_app), [`Gateway`](#gateway), [`GatewayEndpoint`](#gatewayendpoint), [`GatewayEndpoints`](#gatewayendpoints), [`Grant`](#grant), [`KeyRefused`](#keyrefused), [`Keyring`](#keyring), [`Link`](#link), [`Provided`](#provided), [`Refused`](#rollout_traingatewayrefused), [`Reply`](#reply), [`ScoreRequest`](#scorerequest), [`TurnRecord`](#turnrecord), [`turns_table`](#turns_table), [`TurnStore`](#turnstore), [`unaccepted`](#unaccepted)
- **[`rollout_train.jobs`](#rollout_trainjobs)** — A run's job: built from its settings and the cluster config, claiming what it needs. [`driven`](#driven), [`HoursReached`](#hoursreached), [`imitated`](#imitated), [`main`](#main), [`NotEnoughMemory`](#notenoughmemory), [`ran`](#ran), [`Run`](#rollout_trainjobsrun), [`run_directory`](#run_directory), [`SpendReached`](#spendreached), [`taken_by`](#taken_by), [`TrainerActor`](#traineractor), [`TrainerClient`](#trainerclient)
- **[`rollout_train.demand`](#rollout_traindemand)** — What a run's scheduled parts need, and the placement group that reserves them together. [`BRIDGE`](#bridge), [`bridge_asks`](#bridge_asks), [`Bundle`](#bundle), [`colocating`](#colocating), [`Demand`](#demand), [`demand`](#demand), [`HEADROOM`](#headroom), [`Part`](#part), [`placed`](#placed), [`played_channel`](#played_channel), [`Pod`](#rollout_traindemandpod), [`pods`](#pods), [`requested`](#requested), [`reserve`](#reserve), [`Resources`](#resources), [`SUBMITTER`](#submitter), [`TRAINER`](#trainer)
- **[`rollout_train.launching`](#rollout_trainlaunching)** — Asking for a run: its settings in layers, the facts validation reads, the offers. [`capacity_of`](#capacity_of), [`checked`](#checked), [`checkpoints_at`](#checkpoints_at), [`declared`](#declared), [`environment_facts`](#environment_facts), [`Examined`](#examined), [`examined`](#examined), [`free_name`](#free_name), [`ledger_facts`](#ledger_facts), [`offers`](#offers), [`ray_free`](#ray_free), [`Refused`](#rollout_trainlaunchingrefused), [`settled`](#settled)
- **[`rollout_train.submitting`](#rollout_trainsubmitting)** — Starting a run's job as a Ray job or a RayJob, and reading how it goes. [`ask`](#ask), [`Backend`](#backend), [`backend_of`](#backend_of), [`demand_of`](#demand_of), [`entrypoint_of`](#entrypoint_of), [`followed`](#followed), [`job_name`](#job_name), [`JobState`](#jobstate), [`KubernetesApi`](#kubernetesapi), [`POD_SECURITY`](#pod_security), [`pod_security`](#pod_security), [`RayJobResources`](#rayjobresources), [`RayJobs`](#rayjobs), [`rendered`](#rollout_trainsubmittingrendered), [`runtime_env_of`](#rollout_trainsubmittingruntime_env_of), [`sized`](#sized), [`start`](#start), [`stopped`](#stopped), [`submit`](#submit)
- **[`rollout_train.launches`](#rollout_trainlaunches)** — Runs asked for, the jobs they became, and how each goes. [`as_launch`](#as_launch), [`Asked`](#asked), [`changed`](#changed), [`FileLaunches`](#filelaunches), [`Launch`](#launch), [`launch_of`](#launch_of), [`Launches`](#launches), [`launches_of`](#launches_of), [`MOVES`](#moves), [`new_launch`](#new_launch), [`OPEN`](#open), [`stored`](#rollout_trainlaunchesstored)
- **[`rollout_train.monitor`](#rollout_trainmonitor)** — A live web page over every run of a ledger. [`FeedReader`](#feedreader), [`plain`](#plain), [`RunFeed`](#runfeed), [`System`](#system)
- **[`rollout_train.pods`](#rollout_trainpods)** — GPU pods elsewhere: identities, leases, a run's pods, the reaper, the trainer's client. [`GATEWAY_IDENTITY`](#gateway_identity), [`HELD`](#held), [`IDLE`](#idle), [`LeaseLost`](#leaselost), [`live`](#live), [`LivePod`](#livepod), [`needs_of`](#needs_of), [`pod_identity`](#pod_identity), [`pod_leases_of`](#pod_leases_of), [`PodLease`](#podlease), [`PodLeases`](#podleases), [`PodNeed`](#podneed), [`Pods`](#pods), [`PodsDidNotStart`](#podsdidnotstart), [`PodTime`](#podtime), [`reap`](#reap), [`RemoteTrainer`](#remotetrainer), [`STARTING`](#starting), [`TrainerBusy`](#trainerbusy), [`TrainerRefused`](#trainerrefused), [`TrainerUnreachable`](#trainerunreachable)
- **[`rollout_train.pki`](#rollout_trainpki)** — The certificates the platform holds, from the cluster's step-ca, published as Secrets. [`provisioner_key`](#provisioner_key), [`publish`](#rollout_trainpkipublish)
- **[`rollout_train.cluster`](#rollout_traincluster)** — The cluster config: infrastructure, found, read strictly, with secrets only by name. [`auth_problem`](#auth_problem), [`BlobsSection`](#blobssection), [`BridgeSection`](#bridgesection), [`CapacitySection`](#capacitysection), [`Cluster`](#cluster), [`ClusterError`](#clustererror), [`EnvironmentSection`](#environmentsection), [`find`](#find), [`GatewaySection`](#gatewaysection), [`GuardsSection`](#guardssection), [`inspect`](#inspect), [`KubernetesSection`](#kubernetessection), [`LedgerSection`](#ledgersection), [`load`](#load), [`located`](#located), [`MonitorSection`](#monitorsection), [`parsed`](#rollout_trainclusterparsed), [`RaySection`](#raysection), [`RunnersSection`](#runnerssection), [`SandboxesSection`](#sandboxessection), [`ToolsSection`](#toolssection)
- **[`rollout_train.providers`](#rollout_trainproviders)** — Inference providers and trainers: kinds, capabilities, auth, allocation, routing. [`ALLOCATIONS`](#allocations), [`Auth`](#auth), [`AUTHS`](#auths), [`Capabilities`](#capabilities), [`INFERENCE_KINDS`](#inference_kinds), [`InferenceKind`](#inferencekind), [`InferenceProvider`](#inferenceprovider), [`is_local`](#is_local), [`ModelOffer`](#modeloffer), [`POD_FIELDS`](#pod_fields), [`pod_table`](#pod_table), [`PodTable`](#podtable), [`ROUTING`](#routing), [`Routing`](#routing), [`RUNPOD`](#runpod), [`Secret`](#secret), [`settings_of`](#settings_of), [`SettingSpec`](#settingspec), [`Tls`](#tls), [`TRAINER_KINDS`](#trainer_kinds), [`TrainerCapabilities`](#trainercapabilities), [`TrainerKind`](#trainerkind), [`TrainerProvider`](#trainerprovider)
- **[`rollout_train.bridges`](#rollout_trainbridges)** — Bridges between checkpoint formats: the registry, paths, refused pairs, their tasks. [`Bridge`](#bridge), [`bridge_of`](#bridge_of), [`BRIDGED`](#bridged), [`bridged`](#bridged), [`BRIDGES`](#bridges), [`BRIDGING`](#bridging), [`by_name`](#by_name), [`checkpoint_of`](#checkpoint_of), [`Context`](#context), [`format_of`](#format_of), [`FORMATS`](#formats), [`key`](#key), [`made`](#made), [`NoBridge`](#nobridge), [`on_ray`](#on_ray), [`path`](#path), [`rank_factor`](#rank_factor), [`REFUSED`](#refused), [`verbatim`](#verbatim)
- **[`rollout_train.objectives`](#rollout_trainobjectives)** — Objectives declared: families, components, presets, and resolving them. [`Advantage`](#advantage), [`Clip`](#clip), [`Component`](#component), [`component`](#component), [`COMPONENTS`](#components), [`composed`](#composed), [`DEFAULT`](#default), [`Distillation`](#distillation), [`Entropy`](#entropy), [`FAMILIES`](#families), [`from_trainer_settings`](#from_trainer_settings), [`Importance`](#importance), [`Kl`](#kl), [`LEGACY`](#legacy), [`Likelihood`](#likelihood), [`Objective`](#objective), [`objective_of`](#objective_of), [`Preference`](#preference), [`Preset`](#rollout_trainobjectivespreset), [`PRESETS`](#presets), [`problems`](#rollout_trainobjectivesproblems), [`resolved`](#resolved)
- **[`rollout_train.run_settings`](#rollout_trainrun_settings)** — A run's settings: the schema, layers, flags and files, a full copy, diffs. [`Change`](#change), [`diff`](#diff), [`flattened`](#flattened), [`from_file`](#from_file), [`from_flags`](#from_flags), [`is_trainers`](#is_trainers), [`Key`](#key), [`key_of`](#key_of), [`KEYS`](#keys), [`KINDS`](#kinds), [`layered`](#layered), [`objective_in`](#objective_in), [`recorded`](#recorded), [`RunSettings`](#runsettings), [`shortcuts`](#shortcuts), [`WEIGHTS`](#weights)
- **[`rollout_train.stores`](#rollout_trainstores)** — The ledger and the blob store a cluster config names, opened on this node. [`blobs_at`](#blobs_at), [`cluster_ledger`](#cluster_ledger), [`described`](#described), [`FILES`](#files), [`for_pods`](#for_pods), [`ledger_at`](#ledger_at), [`ledger_of`](#ledger_of), [`ledger_url`](#ledger_url), [`location`](#location), [`opened`](#opened), [`opened_ledger`](#opened_ledger), [`store_named`](#store_named), [`Stores`](#stores)
- **[`rollout_train.ledger_service`](#rollout_trainledger_service)** — The ledger over HTTP: the service, the client every role can use, pods' tokens. [`app`](#app), [`Conflict`](#rollout_trainledger_serviceconflict), [`Forbidden`](#forbidden), [`HttpLedger`](#httpledger), [`LedgerUnreachable`](#ledgerunreachable), [`PLATFORM`](#platform), [`pod_token`](#pod_token), [`Scope`](#scope), [`scope_of`](#scope_of)
- **[`rollout_train.presets`](#rollout_trainpresets)** — Named, versioned run settings beside the ledger. [`DatabasePresets`](#databasepresets), [`FilePresets`](#filepresets), [`parsed`](#rollout_trainpresetsparsed), [`Preset`](#rollout_trainpresetspreset), [`Presets`](#presets), [`presets_of`](#presets_of)
- **[`rollout_train.published`](#rollout_trainpublished)** — Versions of environments imported from their source, beside the ledger. [`DatabaseEnvironmentVersions`](#databaseenvironmentversions), [`environment_versions_of`](#environment_versions_of), [`EnvironmentVersion`](#environmentversion), [`EnvironmentVersions`](#environmentversions), [`FileEnvironmentVersions`](#fileenvironmentversions), [`is_published`](#is_published), [`loaded`](#rollout_trainpublishedloaded), [`parsed`](#rollout_trainpublishedparsed), [`provenance`](#provenance), [`short`](#short)
- **[`rollout_train.publishing`](#rollout_trainpublishing)** — Importing an environment from git: fetched, stored, checked on Ray, recorded. [`checked_on_ray`](#checked_on_ray), [`checked_with`](#checked_with), [`entry_point_of`](#entry_point_of), [`EXCLUDED`](#excluded), [`fetched`](#fetched), [`GROUP`](#group), [`Importer`](#importer), [`MARK`](#mark), [`missing`](#missing), [`packed`](#rollout_trainpublishingpacked), [`Project`](#project), [`project_of`](#project_of), [`publish`](#rollout_trainpublishingpublish), [`Published`](#published), [`Refused`](#rollout_trainpublishingrefused), [`report`](#report), [`runtime_env_of`](#rollout_trainpublishingruntime_env_of), [`Source`](#source), [`stored`](#rollout_trainpublishingstored)
- **[`rollout_train.validation`](#rollout_trainvalidation)** — One pure check of a run's settings against a cluster, with its rule table. [`check`](#check), [`CheckpointFacts`](#checkpointfacts), [`completed`](#completed), [`EnvironmentFacts`](#environmentfacts), [`estimated_spend`](#estimated_spend), [`Finding`](#finding), [`LedgerFacts`](#ledgerfacts), [`refusals`](#refusals), [`renderers_of`](#renderers_of), [`Rule`](#rule), [`RULES`](#rules), [`serves`](#serves), [`Spend`](#spend), [`spend_of`](#spend_of), [`SuiteEntryFacts`](#suiteentryfacts), [`SuiteFacts`](#suitefacts), [`weights_of`](#weights_of), [`with_renderers`](#with_renderers), [`with_weights`](#with_weights)
- **[`rollout_train.memory`](#rollout_trainmemory)** — What a trainer needs of each GPU's memory, from the model's files and its GPUs. [`ALLOWANCE_GIB`](#allowance_gib), [`GPU_MEMORY_GIB`](#gpu_memory_gib), [`gpu_memory_gib`](#gpu_memory_gib), [`holds_whole_base`](#holds_whole_base), [`model_facts`](#model_facts), [`ModelFacts`](#modelfacts), [`SEGMENT_TOKENS`](#segment_tokens), [`trainer_memory`](#trainer_memory), [`TrainerMemory`](#trainermemory)
- **[`rollout_train.slots`](#rollout_trainslots)** — A program's model slots bound to a run's channels, and the bindings a run may not make. [`bound`](#bound), [`Declared`](#declared), [`problems`](#rollout_trainslotsproblems), [`serving`](#serving), [`subject`](#subject)
- **[`rollout_train.testing`](#rollout_traintesting)** — Test doubles: a scripted engine and a readable token format. [`admitted`](#admitted), [`Characters`](#characters), [`gateway_endpoints`](#gateway_endpoints), [`keyring`](#keyring), [`LEDGER_TOKEN`](#ledger_token), [`plain_channel`](#plain_channel), [`plain_renderer`](#plain_renderer), [`PlainRenderer`](#plainrenderer), [`Policy`](#policy), [`sample_request`](#sample_request), [`scripted_engine`](#scripted_engine), [`scripted_top`](#scripted_top), [`ScriptedEngine`](#scriptedengine), [`ScriptedTrainer`](#scriptedtrainer), [`SECRETS`](#secrets), [`served_ledger`](#served_ledger)
- **[`rollout_vllm`](#rollout_vllm)** — An engine on vLLM. [`VllmEngine`](#vllmengine)
- **[`rollout_lora`](#rollout_lora)** — A trainer for 4-bit checkpoints with LoRA. [`FullTrainer`](#fulltrainer), [`LoraSettings`](#lorasettings), [`LoraTrainer`](#loratrainer)
- **[`rollout_objectives.settings`](#rollout_objectivessettings)** — A policy step's settings, which the LoRA, full-weight and Tinker trainers take. [`CHANGEABLE`](#changeable), [`OBJECTIVE`](#objective), [`StepSettings`](#stepsettings)
- **[`rollout_objectives.terms`](#rollout_objectivesterms)** — An objective's loss composed from its components, in torch. [`importance_weight`](#importance_weight), [`kl_estimate`](#kl_estimate), [`labelled`](#labelled), [`likelihood`](#likelihood), [`moved_kl`](#moved_kl), [`pair`](#pair), [`policy_gradient`](#policy_gradient), [`reduced`](#reduced), [`Scored`](#scored), [`SUMS`](#sums), [`TALLIED`](#tallied), [`tally`](#tally), [`Terms`](#terms), [`terms`](#terms), [`units`](#units)
- **[`rollout_objectives.step`](#rollout_objectivesstep)** — A step over a batch on a local policy, its plan of minibatches, and its statistics. [`line`](#line), [`metrics`](#metrics), [`MINIBATCHES`](#minibatches), [`minibatches`](#minibatches), [`PackingPolicy`](#packingpolicy), [`Plan`](#rollout_objectivesstepplan), [`PolicyStep`](#policystep), [`preference_terms`](#preference_terms), [`sampled`](#sampled), [`SharedPolicy`](#sharedpolicy), [`TrainablePolicy`](#trainablepolicy)
- **[`rollout_objectives.packing`](#rollout_objectivespacking)** — Many segments in one row of a model's input, a prefix several share once. [`Group`](#group), [`grouped`](#grouped), [`Pack`](#pack), [`packed`](#rollout_objectivespackingpacked), [`packs`](#packs), [`Run`](#rollout_objectivespackingrun), [`sampled_positions`](#sampled_positions), [`Scores`](#rollout_objectivespackingscores), [`SHARED_PREFIX`](#shared_prefix)
- **[`rollout_objectives.ranks`](#rollout_objectivesranks)** — The processes a step is shared among, one per GPU, and how a minibatch is shared. [`Ranks`](#ranks), [`shares`](#shares)
- **[`rollout_qwen`](#rollout_qwen)** — Renderers for the Qwen model families. [`qwen3`](#qwen3), [`qwen35`](#qwen35)
- **[`rollout_gemma`](#rollout_gemma)** — Renderers for the Gemma model families. [`arguments`](#arguments), [`gemma4`](#gemma4), [`GemmaFunctionCalls`](#gemmafunctioncalls)
- **[`rollout_openai`](#rollout_openai)** — A model endpoint for the OpenAI Responses API, on an API key or a Codex login. [`ApiKey`](#apikey), [`codex_provider`](#codex_provider), [`CodexLogin`](#codexlogin), [`Credentials`](#credentials), [`hosted`](#rollout_openaihosted), [`ResponsesContract`](#responsescontract), [`ResponsesEndpoint`](#responsesendpoint)
- **[`rollout_anthropic`](#rollout_anthropic)** — A model endpoint for Anthropic's Messages API. [`hosted`](#rollout_anthropichosted), [`MessagesEndpoint`](#messagesendpoint), [`MessagesOptions`](#messagesoptions)
- **[`rollout_s3`](#rollout_s3)** — Blobs in S3 or any S3-compatible object store. [`S3BlobStore`](#s3blobstore)
- **[`rollout_runpod`](#rollout_runpod)** — GPU pods on RunPod, and certificates for them from step-ca. [`decrypted_key`](#decrypted_key), [`fingerprint`](#fingerprint), [`Pod`](#rollout_runpodpod), [`PodSpec`](#podspec), [`RunPod`](#runpod), [`RunPodError`](#runpoderror), [`StepCa`](#stepca)
- **[`rollout_tinker`](#rollout_tinker)** — A trainer and an engine at Thinking Machines (Tinker). [`TinkerEngine`](#tinkerengine), [`TinkerSettings`](#tinkersettings), [`TinkerTrainer`](#tinkertrainer)

## `rollout.harness`

Writing tasks, agents and programs; runners; memory; tool sets.

### `Agent`

*class* · `libraries/rollout/src/rollout/harness/agent.py`

```python
class Agent
```

The policy side of the loop: what the model sees each turn, and how its output becomes one action.

The default agent samples the policy slot once per turn with the whole history. Subclass it to change the
context (compaction, windows) or the way it acts (plan-then-act, self-critique).

| Field | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str \| None` | `None` | Sent first in every context when set. |

**Methods**

- `def __init__(self, configuration: Any = None) -> None` — Once per run, with the configuration of its program reference.
- `def select_context(self, history: History, hints: ContextHints) -> list[Message]` — What the model sees this turn. Default: the system prompt and the history shaped by the task's hints.
- `async def act(self, run: RunContext, history: History, tools: list[ToolSpecification]) -> Message` — Produce one action. Default: one sample of the policy slot.

### `agent_program`

*function* · `libraries/rollout/src/rollout/harness/runner.py`

```python
def agent_program(task: type[Task], agent: type[Agent] = Agent, *, task_parameters: JsonValue = None, agent_configuration: JsonValue = None) -> ProgramReference
```

A reference to the task loop for `task` and `agent`.

### `AgentProgram`

*class* · `libraries/rollout/src/rollout/harness/program.py`

```python
class AgentProgram(Program)
```

The task loop: a task and an agent.

**Methods**

- `def __init__(self, task: Task, agent: Agent) -> None`
- `def model_slots(self) -> Mapping[str, ModelSlot]`
- `def context_hints(self) -> ContextHints`
- `def imports(self) -> list[str]`
- `def sandboxes(self) -> Mapping[str, SandboxSpec]`
- `def tool_specifications(self) -> list[ToolSpecification]`
- `async def main(self, run: RunContext) -> None`

### `bind`

*function* · `libraries/rollout/src/rollout/harness/runner.py`

```python
def bind(reference: ProgramReference, channel: str, *, slots: Mapping[str, str] | None = None, tools: Mapping[str, ToolBinding] | None = None, pools: Mapping[str, PoolBinding] | None = None) -> RunBinding
```

A binding that serves each model slot of a program from the recorded channel `slots` names for it, else from
`channel` (each recorded as trained or not, as the slot declares), each of its imports from the tool set registered
under the import's own name (or as `tools` says), and each kind of sandbox it declares from the pool registered
under the kind's name (or as `pools` says).

### `Blobs`

*class* · `libraries/rollout/src/rollout/harness/blobs.py`

```python
class Blobs(Protocol)
```

**Methods**

- `async def put(self, data: bytes, media_type: str) -> BlobReference` — Store bytes, or find them already stored; either way return their reference. Either way the blob's time
  is now: it is not deleted for a while (`delete`).
- `async def read(self, reference: BlobReference) -> bytes`
- `async def delete(self, reference: BlobReference, *, unused_for: float = 0.0) -> None` — Remove a blob if it is there, and was not put (written or found) in the last `unused_for` seconds.
  Whoever stored the same bytes holds the same blob: delete only what nothing else names, and with `unused_for`
  longer than any writer takes from putting a blob to naming it.

### `Capacity`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Capacity(ContractModel)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `size` | `int` | required | How many sandboxes the pool can hold at once. |
| `leased` | `int` | required | How many it holds, or is making, now. |

**Methods**

- `@property def free(self) -> int`
- `def to_json(self) -> dict[str, JsonValue]`

### `CompactingAgent`

*class* · `libraries/rollout/src/rollout/harness/memory.py`

```python
class CompactingAgent(Agent)
```

The default agent, with a memory that fits: one sample of the policy slot per turn, over the recent turns and
the agent's own summary of the older ones.

| Field | Type | Default | Description |
|---|---|---|---|
| `compact_prompt` | `str` | `PROMPT` |  |

**Methods**

- `def __init__(self, configuration: object = None) -> None`
- `async def act(self, run: RunContext, history: History, tools: list[ToolSpecification]) -> Message`

### `ContextHints`

*class* · `libraries/rollout/src/rollout/harness/history.py`

```python
class ContextHints
```

Advisory for the agent: how much history the task needs the model to see.

| Field | Type | Default | Description |
|---|---|---|---|
| `history` | `HistoryShape` | `HistoryShape.FULL` |  |
| `window` | `int \| None` | `None` | For `WINDOW`: the number of recent turns. |

### `DeduplicatingToolSet`

*class* · `libraries/rollout/src/rollout/harness/imports.py`

```python
class DeduplicatingToolSet(ToolSet, Protocol)
```

A tool set that says whether it performs each `effect_id` at most once (a `deduplicates = True` attribute, on
a class): its side-effecting tools are then safe to call again (docs/libraries/rollout/contracts/effects.md).

**Methods**

- `@property def deduplicates(self) -> bool`

### `DirectModel`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class DirectModel(ContractModel)
```

A model served by a provider's API through a direct adapter; nothing is recorded.

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | required | The key of an endpoint factory registered with the runner, e.g. `codex`. |
| `model` | `str` | required |  |
| `sampling` | `SamplingParameters` | `SamplingParameters()` |  |

### `Effects`

*class* · `libraries/rollout/src/rollout/harness/model.py`

```python
class Effects(Protocol)
```

How a run performs effects: the run context gives each its identity, records it and performs it.

**Methods**

- `async def perform[T](self, kind: EffectKind, arguments: JsonValue, execute: Callable[[str, str], Awaitable[T]], *, completion: Callable[[T], JsonValue]) -> T` — Assign the next `effect_id`, digest `arguments`, and run `execute(effect_id, arguments_digest)`.
  `completion` renders the result for the run's events.

### `End`

*function* · `libraries/rollout/src/rollout/harness/observation.py`

```python
def End(reward: float | None = None, *, truncated: bool = False, info: Mapping[str, Any] | None = None) -> Observation
```

A terminal observation.

### `Ending`

*class* · `libraries/rollout/src/rollout/harness/observation.py`

```python
class Ending(StrEnum)
```

How an episode ended.

| Member | Value | Description |
|---|---|---|
| `TERMINATED` | `'terminated'` | A real end state (value methods do not bootstrap). |
| `TRUNCATED` | `'truncated'` | Stopped by a limit: turns, time, budget (value methods may bootstrap). |

### `EndpointModel`

*class* · `libraries/rollout/src/rollout/harness/model.py`

```python
class EndpointModel
```

A model slot bound to an endpoint. Every sample sends the full context.

**Methods**

- `def __init__(self, endpoint: ModelEndpoint, session_id: str, effects: Effects, *, retries: int = 3, backoff: float = 1.0) -> None`
- `@property def capabilities(self) -> CapabilityContract`
- `@property def usage(self) -> Usage | None`
- `def address(self) -> ModelAddress`
- `async def sample(self, messages: Sequence[Message], *, tools: Sequence[ToolSpecification] = (), max_output_tokens: int | None = None, tool_choice: ToolChoice | None = None, links: Sequence[SampleLink] = ()) -> Message`

### `FileBlobStore`

*class* · `libraries/rollout/src/rollout/harness/blobs.py`

```python
class FileBlobStore
```

Implements `Blobs` in a directory: one file per blob, named by its SHA-256. A blob's time is its file's
modification time: a put that finds the file sets it to now. Deleting with `unused_for` moves the file aside first
and looks at its time again there, so a put that found it just before is seen (the file is put back), and a put
just after finds no file and writes it again.

**Methods**

- `def __init__(self, directory: Path) -> None`
- `async def put(self, data: bytes, media_type: str) -> BlobReference`
- `async def read(self, reference: BlobReference) -> bytes`
- `async def delete(self, reference: BlobReference, *, unused_for: float = 0.0) -> None`
- `async def put_file(self, path: Path, media_type: str) -> BlobReference` — Store the file at `path`, or find it already stored, without copying its bytes where the store is on the
  same filesystem: the blob is then a hard link to the file, and both are made read-only, since they are one file
  (and share one modification time: the put's). Elsewhere the file is copied. The file is read in pieces, never
  whole.
- `async def link(self, reference: BlobReference, target: Path) -> bool` — Put the blob at `target`: a hard link to it where `target` is on the store's filesystem, else a copy.
  Returns False, putting nothing, if the store does not have it. Raises if the blob is not the one `reference`
  names: its size is checked, and its hash too where it is small (`CHECKED`).

### `History`

*class* · `libraries/rollout/src/rollout/harness/history.py`

```python
class History
```

Read-only for agents and tasks; the loop appends to it through the run context.

**Methods**

- `def __init__(self) -> None`
- `@property def turns(self) -> tuple[Turn, ...]` — Every turn, the start observation first.
- `def append(self, turn: Turn) -> None` — For run contexts only.
- `def messages(self, hints: ContextHints | None = None) -> list[Message]` — The episode's messages in order, shaped by `hints`.

### `HistoryShape`

*class* · `libraries/rollout/src/rollout/harness/history.py`

```python
class HistoryShape(StrEnum)
```

How much of the history the model should see.

| Member | Value | Description |
|---|---|---|
| `FULL` | `'full'` | Everything. |
| `LATEST_OBSERVATION` | `'latest_observation'` | Only the latest observation: observations are complete states. |
| `WINDOW` | `'window'` | The start observation and the most recent `window` turns. |

### `instantiate`

*function* · `libraries/rollout/src/rollout/harness/runner.py`

```python
def instantiate(reference: ProgramReference) -> Program
```

Create the program a reference names.

### `InvalidObservation`

*class* · `libraries/rollout/src/rollout/harness/observation.py`

```python
class InvalidObservation(Exception)
```

A hook returned an observation that breaks the validation rules; the run fails with `INVALID_OBSERVATION`.

### `Lease`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Lease(ContractModel)
```

A sandbox held under a key: what a pool hands out, and what its `Leases` table keeps.

| Field | Type | Default | Description |
|---|---|---|---|
| `key` | `str` | required | What it was acquired under: a run's lease and the sandbox's name (`RUN/GROUP/EPISODE/ATTEMPT/world` for an episode, whose claim it ends with). |
| `kind` | `str` | required |  |
| `pool` | `str` | required | The pool that holds it. |
| `handle` | `str` | required | The pool's name for the sandbox. |
| `addresses` | `Mapping[str, str]` | `Field(default_factory=dict[str, str])` |  |
| `environment` | `Mapping[str, str]` | `Field(default_factory=dict[str, str])` |  |
| `at` | `float` | `0.0` | When it was made, in seconds since the epoch. |
| `seconds` | `float \| None` | `None` | How long it may last from when it was made (`SandboxLimits.seconds`). |
| `lost` | `bool` | `False` | Its sandbox is gone (it ended with the pool's process, say): the key cannot have it back. |

### `LeaseRefused`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class LeaseRefused(Exception)
```

The key may hold no lease now: the claim it was acquired under no longer holds.

### `Leases`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Leases(Protocol)
```

Where a pool keeps its leases, by key: ordinary state, changed in place.

**Methods**

- `async def get(self, key: str) -> Lease | None`
- `async def put(self, lease: Lease) -> None`
- `async def delete(self, key: str) -> None`
- `async def all(self) -> list[Lease]`

### `Memory`

*class* · `libraries/rollout/src/rollout/harness/memory.py`

```python
class Memory
```

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | `str` | `PROMPT` | What the agent is asked when its oldest turns are compacted into a summary. |
| `remembered` | `str` | `REMEMBERED` | How the summary is shown to it afterwards. |
| `summary` | `str` | `''` |  |
| `turns` | `list[list[Message]]` | `field(default_factory=list[list[Message]])` | Its recent turns, oldest first: each a few messages (what it saw, what it replied, how that went). |
| `compactions` | `int` | `0` |  |

**Methods**

- `def context(self, system: Message | None = None, current: Sequence[Message] = ()) -> list[Message]` — The system prompt, the summary, the remembered turns, and what is in front of the agent now.
- `def remember(self, *messages: Message) -> None` — Add a turn.
- `def answer(self, result: str, *, others: str = 'Not done: only your first call of a turn counts.') -> None` — Close the latest turn with how its reply went: the result of its first tool call (further calls are
  answered with `others`), or, for a reply that called nothing, a note to the agent.
- `def crowded(self, model: Model) -> bool` — Whether one more turn might leave the model less than its full room to reply (by what its last prompt
  took, which the model reports, and by how much a turn has been seen to add). The room kept is the contract's
  most output, up to a quarter of the context: a model with no output budget may reply up to its whole context,
  which no compaction could keep free.
- `async def compact(self, model: Model, system: Message | None = None, *, keep: int | None = None) -> None` — Replace the oldest turns with what the agent says it needs to remember of them. The newest `keep` stay as
  they are (by default the newest third).
- `async def sample(self, model: Model, *, system: Message | None = None, current: Sequence[Message] = (), tools: Sequence[ToolSpecification] = (), keep: int = 0) -> Message` — One reply to the context. If the model refuses the context as too long, memory is compacted and the reply
  asked for again (the newest `keep` turns are never compacted).

### `MemoryLeases`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class MemoryLeases
```

`Leases` in this process: they end with it.

**Methods**

- `def __init__(self) -> None`
- `async def get(self, key: str) -> Lease | None`
- `async def put(self, lease: Lease) -> None`
- `async def delete(self, key: str) -> None`
- `async def all(self) -> list[Lease]`

### `Model`

*class* · `libraries/rollout/src/rollout/harness/context.py`

```python
class Model(Protocol)
```

A model slot as code sees it. Nothing here identifies the policy, weights or engine.

**Methods**

- `@property def capabilities(self) -> CapabilityContract`
- `@property def usage(self) -> Usage | None` — Usage reported by the latest sample, if any: drives compaction decisions.
- `async def sample(self, messages: Sequence[Message], *, tools: Sequence[ToolSpecification] = (), max_output_tokens: int | None = None, tool_choice: ToolChoice | None = None, links: Sequence[SampleLink] = ()) -> Message` — A reply to `messages`; `links` say how the request follows from earlier ones of the slot (by the effect
  ids its replies carry: `rollout.harness.model.EFFECT_ID_META`).
- `def address(self) -> ModelAddress` — For a harness that brings its own loop (a coding agent running inside the environment, say): where it
  reaches this slot's model. Hand it the base URL and key; what it samples there is this slot's, recorded like
  any other sample. Raises if the deployment does not serve models over HTTP.

### `ModelBinding`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class ModelBinding(ContractModel)
```

Exactly one of `direct` or `recorded`.

| Field | Type | Default | Description |
|---|---|---|---|
| `direct` | `DirectModel \| None` | `None` |  |
| `recorded` | `RecordedModel \| None` | `None` |  |

### `ModelSample`

*class* · `libraries/rollout/src/rollout/harness/hooks.py`

```python
class ModelSample
```

One model sample: what a slot's model was sent and what it replied.

| Field | Type | Default | Description |
|---|---|---|---|
| `run_id` | `str` | required |  |
| `slot` | `str` | required |  |
| `request` | `SampleRequest` | required | `request.context.append` holds the messages sent; `request.tools` the tools offered. |
| `result` | `SampleResult` | required |  |
| `seconds` | `float` | required | How long the endpoint took. |

### `ModelSlot`

*class* · `libraries/rollout/src/rollout/harness/task.py`

```python
class ModelSlot
```

A model the task declares. The agent acts through `policy`; other slots (a simulated user, an opponent, a
judge) are sampled by the task itself. What serves each slot is the run's binding's to say.

| Field | Type | Default | Description |
|---|---|---|---|
| `trained` | `bool` | `True` | Whether a run may train on the slot's turns. A slot that is not (a judge, a fixed opponent) is recorded like any other, and a run binds it to a channel by name (`slots.SLOT`). |
| `judge` | `bool` | `False` | Whether the slot judges what other slots did. A judge is never trained, and a run binds it to a channel that serves the run's own checkpoints only when it says so (`self_judging`). |

### `Mount`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Mount(ContractModel)
```

Files the sandbox sees, read-only: an environment version's files, its virtual environment.

| Field | Type | Default | Description |
|---|---|---|---|
| `source` | `str` | required | Where they are, as the pool finds them: a path on its machine, or a name it resolves. |
| `target` | `str` | required | Where they appear inside the sandbox. |

### `Network`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Network(ContractModel)
```

What the sandbox may reach besides its connection to the runner (its lease's addresses): nothing, unless hosts
are allowed.

| Field | Type | Default | Description |
|---|---|---|---|
| `allow` | `FrozenSequence[str]` | `()` | Hosts it may reach (`pypi.org`, say). |

### `NoCapacity`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class NoCapacity(Exception)
```

The pool holds as many sandboxes as it can; an acquire may succeed once one is released.

### `Observation`

*class* · `libraries/rollout/src/rollout/harness/observation.py`

```python
class Observation
```

What the model is shown next, with the reward for the reply it answers.

`Observation("text")` builds one USER text message; `Observation(message)` and `Observation([messages])` take
canonical messages.

| Field | Type | Default | Description |
|---|---|---|---|
| `messages` | `tuple[Message, ...]` | see constructor | Shown to the model next: USER and TOOL messages only. |
| `reward` | `float \| None` | see constructor | Bound to the reply this observation answers. |
| `end` | `Ending \| None` | see constructor | `None` means the episode continues. |
| `info` | `Mapping[str, Any]` | see constructor | Logged; never shown to the model. |

**Methods**

- `def __init__(self, messages: ObservationContent = (), *, reward: float | None = None, end: Ending | None = None, info: Mapping[str, Any] | None = None) -> None`

### `Pool`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Pool(Protocol)
```

Hands out sandboxes of one kind under leases: `SandboxPool`, or one served over HTTP (`RemotePool`). Runners
acquire and release; programs perform operations through `run.sandbox(name)`.

**Methods**

- `@property def deduplicates(self) -> bool` — Whether it performs each operation's `effect_id` at most once.
- `def operations(self) -> Sequence[ToolSpecification]`
- `async def acquire(self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None = None) -> Lease` — The lease of `key`: the one there is, or a new sandbox. Raises `NoCapacity` when the pool is full,
  `LeaseRefused` for a key that may hold no lease now (its claim lapsed), and `SandboxLost` for a key whose
  sandbox is gone.
- `async def release(self, key: str) -> None` — End the lease of `key` and delete its sandbox; nothing if there is no such lease.
- `async def capacity(self) -> Capacity`
- `async def call(self, key: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str) -> ToolResult` — Perform an operation on the sandbox leased under `key`.

### `PoolBinding`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class PoolBinding(ContractModel)
```

How a kind of sandbox is served. Exactly one kind is set.

| Field | Type | Default | Description |
|---|---|---|---|
| `local` | `str \| None` | `None` | The name of a pool registered with the runner, in process. |
| `url` | `str \| None` | `None` | A pool served over HTTP (`rollout.harness.remote.serve_pool`), wherever its sandboxes live. |

### `Process`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Process(ContractModel)
```

A process the sandbox runs from its start (an environment's worker, a coding agent): the pool launches it.

| Field | Type | Default | Description |
|---|---|---|---|
| `command` | `FrozenSequence[str]` | required |  |
| `environment` | `Mapping[str, str]` | `Field(default_factory=dict[str, str])` | Its own variables; the lease's (its slots' model addresses) are added to them. |
| `directory` | `str \| None` | `None` | Its working directory, inside the sandbox. |

### `Program`

*class* · `libraries/rollout/src/rollout/harness/program.py`

```python
class Program
```

What a run executes. `AgentProgram` is the task loop; other programs drive their own.

**Methods**

- `def model_slots(self) -> Mapping[str, ModelSlot]` — The model slots the program samples; a runner binds an endpoint to each.
- `def context_hints(self) -> ContextHints`
- `def imports(self) -> list[str]` — The imported tool sets the program needs; the run's binding says how each is served.
- `def sandboxes(self) -> Mapping[str, SandboxSpec]` — The sandboxes the program runs against, by name: the runner acquires each from the pool the binding names
  for its kind before `main`, and releases it after; the program reaches it as `run.sandbox(name)`.
- `def tool_specifications(self) -> list[ToolSpecification]` — Tools the program itself defines (`@tool` methods), for the run's `tools.resolved` event.
- `async def main(self, run: RunContext) -> None`

### `ProgramReference`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class ProgramReference(ContractModel)
```

What a run executes, by name, so a runner in another process can re-create it.

| Field | Type | Default | Description |
|---|---|---|---|
| `program` | `str` | required | `module:QualifiedName` of a `Program` class. |
| `parameters` | `JsonValue` | `None` |  |

### `Provider`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Provider(Protocol)
```

Makes, deletes and operates sandboxes of one kind: Paper servers, containers, a provider's API. A
`SandboxPool` leases them out.

**Methods**

- `@property def kind(self) -> str`
- `@property def size(self) -> int` — How many sandboxes it can hold at once.
- `def operations(self) -> Sequence[ToolSpecification]` — What can be done to one of its sandboxes (none: a harness inside reaches it by its addresses).
- `async def create(self, handle: str, spec: SandboxSpec, environment: Mapping[str, str]) -> Reach` — Make the sandbox `handle`, giving what runs inside it `environment`, or say how to reach it if it is
  there already.
- `async def delete(self, handle: str) -> None` — Delete the sandbox, or do nothing if it is gone.
- `async def held(self) -> Sequence[str]` — The handles of the sandboxes it has now.
- `async def call(self, handle: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str) -> ToolResult` — Perform an operation on a sandbox. Errors the operation reports are results with `is_error`; exceptions
  are platform failures.

### `Reach`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Reach(ContractModel)
```

How a sandbox is reached, as its provider says once it has made it.

| Field | Type | Default | Description |
|---|---|---|---|
| `addresses` | `Mapping[str, str]` | `Field(default_factory=dict[str, str])` | Where its services listen, by name (`{"game": "127.0.0.1:25565"}`, say). |
| `environment` | `Mapping[str, str]` | `Field(default_factory=dict[str, str])` | Environment variables for what runs inside it: those it was given, and any of the provider's own. |

### `RecordedEndpoints`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class RecordedEndpoints(Protocol)
```

Serves recorded bindings, as runners see it: the gateway (`rollout_train.gateway.GatewayEndpoints`).

**Methods**

- `def endpoint(self, binding: RecordedModel) -> ModelEndpoint`

### `RecordedModel`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class RecordedModel(ContractModel)
```

A channel served through the gateway, which records every sample.

| Field | Type | Default | Description |
|---|---|---|---|
| `channel` | `str` | required |  |
| `sampling` | `SamplingParameters` | `SamplingParameters()` |  |
| `trained` | `bool` | `True` | Whether its turns may be trained on: the slot's declaration (`ModelSlot.trained`). Every turn records it. |

### `register`

*function* · `libraries/rollout/src/rollout/harness/runner.py`

```python
def register(cls: type) -> str
```

Make a class resolvable by name in this process, even if it cannot be imported (e.g. defined in a script).

### `resolve`

*function* · `libraries/rollout/src/rollout/harness/runner.py`

```python
def resolve(name: str) -> type
```

The class a `module:QualifiedName` names: registered in this process, or imported.

### `rollout`

*function* · `libraries/rollout/src/rollout/harness/loop.py`

```python
async def rollout(task: Task, agent: Agent, run: RunContext) -> None
```

Run one episode of `task` with `agent`. Raises `InvalidObservation` or whatever a hook raised.

### `RunBinding`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class RunBinding(ContractModel)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `models` | `Mapping[str, ModelBinding]` | required | Model slot → how it is served. |
| `imports` | `Mapping[str, ToolBinding]` | `Field(default_factory=dict[str, ToolBinding])` | Import name → how the tool set is served. |
| `pools` | `Mapping[str, PoolBinding]` | `Field(default_factory=dict[str, PoolBinding])` | Sandbox kind → the pool its sandboxes are acquired from. |

### `RunContext`

*class* · `libraries/rollout/src/rollout/harness/context.py`

```python
class RunContext(Protocol)
```

Everything task and agent code can reach during a run. Passed to every hook as `run`.

**Methods**

- `@property def run_id(self) -> str`
- `@property def turn(self) -> int` — Completed model turns so far.
- `@property def history(self) -> History`
- `@property def models(self) -> Mapping[str, Model]`
- `@property def model(self) -> Model` — `models["policy"]`.
- `@property def tools(self) -> Tools` — Imported tools; each call is a `tool.call` effect.
- `def sandbox(self, name: str) -> Sandbox` — A sandbox the program declared (`Program.sandboxes()`), acquired for this run: its addresses, its
  environment, its operations. `KeyError` for a name the program did not declare.
- `@property def blobs(self) -> Blobs | None` — Stores bytes such as images for `Media` blocks; None when the runner has no blob store.
- `@property def random(self) -> random.Random` — Seeded from `run_id`.
- `@property def context_hints(self) -> ContextHints`
- `def now(self) -> datetime` — The current time, in UTC: the time the run's events carry.
- `def reward(self, value: float, *, slot: str = 'policy', key: str = 'default') -> None` — Assign a reward to a model slot outside an observation (e.g. to an opponent, or several keyed rewards).
- `def exclude_from_training(self, reason: str) -> None` — Mark the run as unsuitable for training, e.g. after an infrastructure fault that is not the policy's.
- `async def gather[T](self, *awaitables: Awaitable[T]) -> list[T]` — Await concurrently, in order. Equivalent to `asyncio.gather`.
- `async def emit(self, kind: str, payload: JsonValue) -> None` — Output of the run, such as its result: recorded as an `output.emit` effect and an `output.emitted` event.
- `def record(self, observation: Observation, *, reply: Message | None = None) -> None` — Append a turn to the history.

### `RunHandle`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class RunHandle(Protocol)
```

**Methods**

- `@property def run_id(self) -> str`
- `@property def done(self) -> bool`
- `@property def outcome(self) -> RunOutcome | None` — How the run ended; None while it is live.
- `async def result(self) -> RunOutcome` — Wait for the run to end.
- `def events(self, *, from_seq: int = 0) -> AsyncIterator[RunEvent]` — Every event from `from_seq`, then new ones as they are recorded, until the run ends.
- `def recorded_events(self) -> list[RunEvent]` — Every event recorded so far.

### `RunHooks`

*class* · `libraries/rollout/src/rollout/harness/hooks.py`

```python
class RunHooks
```

Subclass and override what you need; pass instances to a runner (`LocalRunner(hooks=[...])`).

**Methods**

- `def on_event(self, event: RunEvent) -> None` — A run event was recorded (of any run of the runner; `event.run_id` says which).
- `def on_sample(self, sample: ModelSample) -> None` — A model replied.

### `Runner`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class Runner(Protocol)
```

**Methods**

- `async def launch(self) -> None` — Make the runner ready: call it once, before anything else that starts or reaches a run.
- `async def close(self) -> None` — Release what the runner holds; it cannot be used afterwards.
- `def run(self, run_id: str) -> RunHandle` — The handle of a run; `KeyError` if the runner does not know it.
- `async def start(self, specification: RunSpecification, *, run_id: str | None = None, labels: Mapping[str, str] | None = None, lease: str | None = None) -> RunHandle` — Start a run. Its sandboxes are acquired under `lease` and each one's name (by default the `run_id`): an
  episode's claim, say, so that they end with it.
- `async def cancel(self, run_id: str, *, reason: str) -> None`

### `RunOutcome`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class RunOutcome(ContractModel)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `status` | `RunStatus` | required |  |
| `failure_class` | `RunFailureClass \| None` | `None` |  |
| `detail` | `str \| None` | `None` |  |

### `RunSpecification`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class RunSpecification(ContractModel)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `program` | `ProgramReference` | required |  |
| `binding` | `RunBinding` | required |  |

### `RunStatus`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class RunStatus(StrEnum)
```

| Member | Value | Description |
|---|---|---|
| `COMPLETED` | `'completed'` |  |
| `FAILED` | `'failed'` |  |
| `CANCELLED` | `'cancelled'` |  |

### `SamplingParameters`

*class* · `libraries/rollout/src/rollout/harness/runner.py`

```python
class SamplingParameters(ContractModel)
```

Configured on bindings, never by task or agent code.

| Field | Type | Default | Description |
|---|---|---|---|
| `temperature` | `float` | `1.0` |  |
| `top_p` | `float` | `1.0` |  |
| `reasoning_effort` | `str \| None` | `None` | For providers with reasoning controls, e.g. `low`, `medium`, `high`. |
| `thinking_tokens` | `int \| None` | `None` | For a recorded channel: tokens of thinking per turn, and of answer after it, in place of the channel's own (an eval's, say); none: the channel's. |
| `answer_tokens` | `int \| None` | `None` |  |

### `Sandbox`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Sandbox
```

A sandbox a run holds, as `run.sandbox(name)` gives it: how to reach it, and its operations, each a
`tool.call` effect.

**Methods**

- `def __init__(self, name: str, lease: Lease, pool: Pool, effects: Effects) -> None`
- `@property def addresses(self) -> Mapping[str, str]`
- `@property def environment(self) -> Mapping[str, str]` — For what runs inside it: those the runner gave it (its slots' model addresses) and the pool's own.
- `def specifications(self) -> list[ToolSpecification]` — Its operations.
- `async def call(self, operation: str, arguments: Mapping[str, JsonValue] | None = None) -> ToolResult` — Perform an operation as a `tool.call` effect.

### `SandboxLimits`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class SandboxLimits(ContractModel)
```

What the sandbox may use. Unset: as much as the pool gives.

| Field | Type | Default | Description |
|---|---|---|---|
| `cpus` | `float \| None` | `None` |  |
| `memory_mib` | `int \| None` | `None` |  |
| `processes` | `int \| None` | `None` |  |
| `seconds` | `float \| None` | `None` | Time from its start: past it, its lease ends and the pool deletes it (measured by the pool's monotonic clock, which a change of the machine's wall clock does not move). |

### `SandboxLost`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class SandboxLost(Exception)
```

The key's sandbox is gone, and a new one would not be the one its run was using.

### `SandboxPool`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class SandboxPool
```

A `Pool` over a `Provider`: at most `provider.size` leases at once, kept in `leases` under the pool's `name`
(by default the provider's kind; several pools sharing a table need names of their own).

**Methods**

- `def __init__(self, provider: Provider, *, name: str | None = None, leases: Leases | None = None, admits: Callable[[str], Awaitable[bool]] | None = None) -> None` — `admits` says whether a key may hold a lease now (beside a ledger: whether its claim holds,
  `rollout_train.sandboxes.admits`); without it, every key may.
- `@property def deduplicates(self) -> bool`
- `def operations(self) -> Sequence[ToolSpecification]`
- `async def acquire(self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None = None) -> Lease` — The lease of `key`, or a new sandbox. Raises `LeaseRefused` for a key `admits` refuses (releasing a lease
  it has), `SandboxLost` for a key whose sandbox is gone, and `NoCapacity` when the pool is full.
- `async def release(self, key: str) -> None`
- `async def capacity(self) -> Capacity`
- `async def call(self, key: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str) -> ToolResult`
- `async def held(self) -> list[Lease]` — This pool's leases, those whose sandboxes are lost included.
- `async def sweep(self, ended: Callable[[Lease], bool] = lambda lease: False) -> list[str]` — Release the leases `ended` says have ended, and those past their time limit; mark lost those whose sandbox
  is gone (the pool's process was started again, say), which their keys cannot have back; and delete the
  sandboxes no lease names. Returns the keys released or marked lost.
- `async def close(self) -> None` — Release every lease the pool holds (deleting its sandboxes), and close the provider.

### `SandboxSpec`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class SandboxSpec(ContractModel)
```

A sandbox a program needs: its kind, what it is made from, and what it may do. A world for an episode needs
only a kind and parameters; a worker for an environment names a process, mounts, scratch, network and limits.

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `str` | required | The kind of sandbox (`minecraft`, say): the binding names the pool that serves each kind. |
| `parameters` | `Mapping[str, JsonValue]` | `Field(default_factory=dict[str, JsonValue])` | What the pool makes it from: a task and its seeds, an image. |
| `slots` | `FrozenSequence[str]` | `()` | Model slots a harness inside the sandbox samples. Each one's address is put in the sandbox's environment: `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `OPENAI_MODEL`, and `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_MODEL` (`harness_environment`), suffixed with the slot's name in capitals (`_AGENT_1`), and unsuffixed too when there is one slot. A key names the run's session of its slot, and stops working once it expires or a newer attempt of its episode takes the episode's fence. |
| `process` | `Process \| None` | `None` |  |
| `mounts` | `FrozenSequence[Mount]` | `()` |  |
| `scratch` | `Scratch \| None` | `None` | Without it, the sandbox writes nowhere. |
| `network` | `Network` | `Network()` |  |
| `limits` | `SandboxLimits` | `SandboxLimits()` |  |

### `Scratch`

*class* · `libraries/rollout/src/rollout/harness/sandboxes.py`

```python
class Scratch(ContractModel)
```

A directory the sandbox may write, empty when it starts.

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | `'/scratch'` |  |
| `mib` | `int` | `1024` | The most it may hold, in MiB. |

### `Task`

*class* · `libraries/rollout/src/rollout/harness/task.py`

```python
class Task
```

The environment an agent acts in: tools, the first observation, responses to replies, and scoring.

Subclass it and implement `start`; override the other hooks as needed. Declarations are class attributes.

| Field | Type | Default | Description |
|---|---|---|---|
| `models` | `ClassVar[dict[str, ModelSlot]]` | `{'policy': ModelSlot()}` | The model slots the task uses. The agent acts through `policy`. |
| `imports` | `ClassVar[list[str]]` | `[]` | External tool sets, bound per run. |
| `sandboxes` | `ClassVar[dict[str, SandboxSpec]]` | `{}` | Sandboxes the task runs against, by name: acquired before `setup`, reached as `run.sandbox(name)`. |
| `max_turns` | `ClassVar[int \| None]` | `None` | The loop truncates the episode after this many model turns. |
| `context_hints` | `ClassVar[ContextHints]` | `ContextHints()` | Advisory for the agent: how much history the model should see. |
| `declared_tools` | `ClassVar[dict[str, DeclaredTool]]` | `{}` | The `@tool` methods of this class, collected when the class is defined. |

**Methods**

- `def __init__(self, parameters: Any = None) -> None` — Once per run, with the row's parameters. Subclasses may define their own signature.
- `async def setup(self, run: RunContext) -> None` — Once per run, before `start`.
- `async def start(self, run: RunContext) -> Observation` — Open the episode.
- `async def respond(self, run: RunContext, reply: Message) -> Observation` — The environment's step. Default: execute the reply's tool calls; end the episode when there are none.
- `async def score(self, run: RunContext) -> float | None` — Episode-level reward, attached to the end of the trajectory.
- `async def teardown(self, run: RunContext) -> None` — Always runs if `setup` began; must be idempotent.
- `def tools_for_turn(self, run: RunContext) -> list[ToolSpecification]` — The tools offered this turn. Default: every `@tool` method and every imported tool.
- `async def run_tools(self, run: RunContext, reply: Message) -> Observation` — Execute every tool call in `reply` concurrently; one TOOL message answers them all.

### `tool`

*function* · `libraries/rollout/src/rollout/harness/tools.py`

```python
def tool[F: Callable[..., Any]](function: F | None = None, /, *, name: str | None = None, retry_class: RetryClass = RetryClass.PURE, timeout: timedelta | None = None) -> F | Callable[[F], F]
```

Declare a task method as a tool: `@tool` or `@tool(name=..., retry_class=..., timeout=...)`.

### `ToolBinding`

*class* · `libraries/rollout/src/rollout/harness/imports.py`

```python
class ToolBinding(ContractModel)
```

How an import is served. Exactly one kind is set.

| Field | Type | Default | Description |
|---|---|---|---|
| `local` | `str \| None` | `None` | The name of a tool set registered with the runner, in process. |
| `url` | `str \| None` | `None` | A tool set served over HTTP (`rollout.harness.remote`): an environment's own infrastructure, wherever it runs. |

### `Tools`

*class* · `libraries/rollout/src/rollout/harness/imports.py`

```python
class Tools
```

The imported tools of a run (`run.tools`).

**Methods**

- `def __init__(self, tool_sets: Mapping[str, ToolSet], effects: Effects) -> None`
- `def specifications(self) -> list[ToolSpecification]`
- `async def call(self, name: str, arguments: Mapping[str, JsonValue]) -> ToolResult` — Call an imported tool as a `tool.call` effect.

### `ToolSet`

*class* · `libraries/rollout/src/rollout/harness/imports.py`

```python
class ToolSet(Protocol)
```

A provider of tools: in process, or a client of an MCP server, an HTTP service or another agent.

**Methods**

- `def specifications(self) -> Sequence[ToolSpecification]`
- `async def call(self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str) -> ToolResult` — Perform one call. Tool-level errors are results with `is_error`; exceptions are platform failures.

### `Turn`

*class* · `libraries/rollout/src/rollout/harness/history.py`

```python
class Turn
```

One step of the episode: a reply and the observation that answers it. The start observation has no reply.

| Field | Type | Default | Description |
|---|---|---|---|
| `reply` | `Message \| None` | required |  |
| `observation` | `Observation` | required |  |

### `with_row`

*function* · `libraries/rollout/src/rollout/harness/runner.py`

```python
def with_row(reference: ProgramReference, row: JsonValue) -> ProgramReference
```

The same program for another row of parameters (for the task loop: the task's parameters).

## `rollout.contracts`

Types that cross layers: canonical content, identifiers, digests, effects, events.

### `address_of`

*function* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
def address_of(endpoint: ModelEndpoint, session_id: str) -> ModelAddress
```

`AddressableEndpoint.address` of an endpoint; raises if the endpoint has no address.

### `AddressableEndpoint`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class AddressableEndpoint(ModelEndpoint, Protocol)
```

A model endpoint that also serves its slots over HTTP, to a harness that brings its own loop.

**Methods**

- `def address(self, session_id: str) -> ModelAddress` — Where such a harness reaches the session's slot.

### `arguments_digest`

*function* · `libraries/rollout/src/rollout/contracts/digests.py`

```python
def arguments_digest(arguments: JsonValue | BaseModel) -> str
```

Sent with every `effect_id`; a tool set refuses a known `effect_id` whose arguments digest differs
(`Conflict`).

### `BlobReference`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class BlobReference(ContractModel)
```

Content kept in a blob store, named by where it is and what it hashes to.

| Field | Type | Default | Description |
|---|---|---|---|
| `uri` | `str` | required |  |
| `sha256` | `str` | required |  |
| `size` | `int` | required |  |
| `media_type` | `str` | required |  |

### `Block`

*type alias* · `libraries/rollout/src/rollout/contracts/content.py`

```python
type Block = Annotated[Text | Media | ToolCall | ToolResultBlock | Reasoning, Field(discriminator='type')]
```

### `canonical_json`

*function* · `libraries/rollout/src/rollout/contracts/digests.py`

```python
def canonical_json(value: JsonValue | BaseModel) -> bytes
```

RFC 8785 canonical JSON; a model is dumped with `None` fields omitted.

### `CapabilityContract`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class CapabilityContract(ContractModel)
```

What a model slot guarantees. It must not weaken during a run.

| Field | Type | Default | Description |
|---|---|---|---|
| `context_limit` | `int` | required | Minimum guaranteed. |
| `max_output_tokens` | `int` | required |  |

### `Conflict` {#rolloutcontractsconflict}

*class* · `libraries/rollout/src/rollout/contracts/effects.py`

```python
class Conflict(Exception)
```

A receiver that deduplicates by `effect_id` was sent a known `effect_id` with a different arguments digest.

### `context_digests`

*function* · `libraries/rollout/src/rollout/contracts/digests.py`

```python
def context_digests(messages: Sequence[Message]) -> list[str]
```

The digest chain `d₀ … dₙ` of a context: `dᵢ = sha256(dᵢ₋₁ ‖ sha256(JCS(itemᵢ)))` over raw digest bytes.

`dₖ` identifies the prefix of length `k`, so a retained prefix is recognizable by its own chain value.

### `ContextDelta`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class ContextDelta(ContractModel)
```

The context of a request: its messages, and the digest that names them.

| Field | Type | Default | Description |
|---|---|---|---|
| `append` | `FrozenSequence[Message]` | `()` | The whole context, in order. |
| `digest` | `str` | required | The last value of the chain `rollout.contracts.digests.context_digests` computes over `append`. |

### `ContextOverflow`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class ContextOverflow(ModelEndpointError)
```

The context exceeds the contract's limit; agents compact and retry.

**Methods**

- `def __init__(self, context_limit: int) -> None`

### `ContractModel`

*class* · `libraries/rollout/src/rollout/contracts/base.py`

```python
class ContractModel(BaseModel)
```

Base for every contract type: immutable, and unknown fields are kept.

Keeping unknown fields lets a component read and re-write a record written by newer code without dropping
what it does not understand.

### `ContractViolation`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class ContractViolation(ModelEndpointError)
```

The request exceeds the capability contract.

### `digest`

*function* · `libraries/rollout/src/rollout/contracts/digests.py`

```python
def digest(value: JsonValue | BaseModel) -> str
```

Lowercase hexadecimal SHA-256 of the canonical JSON.

### `effect_id`

*function* · `libraries/rollout/src/rollout/contracts/identifiers.py`

```python
def effect_id(run_id: str, generation: int, ordinal: int) -> str
```

`{run_id}:{generation}:{ordinal}`: the same on every re-execution, so it is the universal idempotency key.

### `EffectIdentity`

*class* · `libraries/rollout/src/rollout/contracts/identifiers.py`

```python
class EffectIdentity
```

The parts of an `effect_id`: `{run_id}:{generation}:{ordinal}`.

| Field | Type | Default | Description |
|---|---|---|---|
| `run_id` | `str` | required |  |
| `generation` | `int` | required |  |
| `ordinal` | `int` | required |  |

**Methods**

- `@classmethod def parse(cls, effect_id: str) -> 'EffectIdentity'`

### `EffectKind`

*class* · `libraries/rollout/src/rollout/contracts/effects.py`

```python
class EffectKind(StrEnum)
```

The catalog of effects.

| Member | Value | Description |
|---|---|---|
| `MODEL_SAMPLE` | `'model.sample'` |  |
| `TOOL_CALL` | `'tool.call'` |  |
| `OUTPUT_EMIT` | `'output.emit'` |  |

### `EffectStatus`

*class* · `libraries/rollout/src/rollout/contracts/effects.py`

```python
class EffectStatus(StrEnum)
```

How an effect completed.

| Member | Value | Description |
|---|---|---|
| `OK` | `'ok'` |  |
| `FAILED` | `'failed'` |  |

### `EMPTY_DIGEST`

*constant* · `libraries/rollout/src/rollout/contracts/digests.py`

```python
EMPTY_DIGEST = hashlib.sha256(b'').hexdigest()
```

`d₀` of every context digest chain.

### `FinishReason`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class FinishReason(StrEnum)
```

Why a sample stopped.

| Member | Value | Description |
|---|---|---|
| `STOP` | `'stop'` |  |
| `LENGTH` | `'length'` |  |
| `TOOL_USE` | `'tool_use'` |  |

### `FrozenSequence`

*type alias* · `libraries/rollout/src/rollout/contracts/base.py`

```python
type FrozenSequence[T] = Annotated[Sequence[T], AfterValidator(tuple)]
```

A sequence field that accepts any sequence and is stored as a tuple, so contract values stay immutable.

### `InternalError`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class InternalError(ModelEndpointError)
```

The endpoint failed.

### `Media`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class Media(ContractModel)
```

An image, audio clip or document.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `Literal['media']` | `'media'` |  |
| `media_type` | `str` | required |  |
| `source` | `BlobReference` | required |  |

### `Message`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class Message(ContractModel)
```

A message in canonical form: a role and a sequence of content blocks.

TOOL messages contain only `ToolResultBlock`s, and `ToolCall`s appear only in ASSISTANT messages.

| Field | Type | Default | Description |
|---|---|---|---|
| `role` | `Role` | required |  |
| `content` | `FrozenSequence[Block]` | `()` |  |
| `meta` | `Mapping[str, str]` | `Field(default_factory=dict[str, str])` | Not model-visible; never rendered and not covered by the context digest. |

**Methods**

- `@property def text(self) -> str` — The concatenated text blocks.
- `@property def tool_calls(self) -> list[ToolCall]` — The tool calls, in order.
- `@classmethod def user(cls, text: str) -> 'Message'` — A USER message with one text block.
- `@classmethod def assistant(cls, text: str) -> 'Message'` — An ASSISTANT message with one text block.
- `@classmethod def system(cls, text: str) -> 'Message'` — A SYSTEM message with one text block.

### `message_digest`

*function* · `libraries/rollout/src/rollout/contracts/digests.py`

```python
def message_digest(message: Message) -> str
```

Covers what the model can see: `meta` is excluded.

### `ModelAddress`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class ModelAddress(ContractModel)
```

Where a harness that brings its own loop reaches a model slot: an endpoint that speaks OpenAI's and
Anthropic's APIs. Whatever answers there is the slot's model; the harness only sets its base URL and key.

| Field | Type | Default | Description |
|---|---|---|---|
| `base_url` | `str` | required |  |
| `api_key` | `str` | required | Names the session: valid for this run's slot only. |
| `model` | `str` | required | What to send as the model's name (the endpoint ignores it). |

### `ModelEndpoint`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class ModelEndpoint(Protocol)
```

Serves model slots: implemented by the gateway's endpoints and by direct adapters. An endpoint that can also be
reached over HTTP is an `AddressableEndpoint`.

**Methods**

- `def describe(self, session_id: str) -> CapabilityContract` — The capability contract of the session's model slot.
- `async def sample(self, request: SampleRequest) -> SampleResult` — One reply. Where the endpoint deduplicates, a repeated `effect_id` returns the recorded result.
- `async def cancel(self, effect_id: str) -> None` — Best-effort.

### `ModelEndpointError`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class ModelEndpointError(Exception)
```

Errors an endpoint raises; see the table in the contract for how the core handles each.

### `NamedToolChoice`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class NamedToolChoice(ContractModel)
```

The model must call this tool.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |

### `new_run_id`

*function* · `libraries/rollout/src/rollout/contracts/identifiers.py`

```python
def new_run_id() -> str
```

`r_{ulid}`.

### `new_ulid`

*function* · `libraries/rollout/src/rollout/contracts/identifiers.py`

```python
def new_ulid() -> str
```

A ULID: 48 bits of Unix time in milliseconds, then 80 random bits, in Crockford base 32 (26 characters).

Minted by runners and services, never by task code (whose randomness is `run.random`, seeded from the run's id).

### `Overloaded`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class Overloaded(ModelEndpointError)
```

Admission control: retry after `retry_after` seconds.

**Methods**

- `def __init__(self, retry_after: float | None = None) -> None`

### `Reasoning`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class Reasoning(ContractModel)
```

The model's reasoning.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `Literal['reasoning']` | `'reasoning'` |  |
| `scope` | `ReasoningScope` | required |  |
| `text` | `str` | required |  |

### `ReasoningScope`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class ReasoningScope(StrEnum)
```

Who can consume a reasoning block.

| Member | Value | Description |
|---|---|---|
| `PORTABLE` | `'portable'` | Plain text that any renderer may render or drop. |

### `ResultBlock`

*type alias* · `libraries/rollout/src/rollout/contracts/content.py`

```python
type ResultBlock = Annotated[Text | Media, Field(discriminator='type')]
```

### `RetryClass`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class RetryClass(StrEnum)
```

Whether a tool call is safe to perform again with the same `effect_id` (docs/guide/tools.md#retry-classes).

| Member | Value | Description |
|---|---|---|
| `PURE` | `'pure'` |  |
| `IDEMPOTENT` | `'idempotent'` |  |
| `SIDE_EFFECTING` | `'side_effecting'` |  |
| `UNKNOWN` | `'unknown'` |  |

### `Role`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class Role(StrEnum)
```

Who a message is from. Observations contain only USER and TOOL messages.

| Member | Value | Description |
|---|---|---|
| `SYSTEM` | `'system'` |  |
| `USER` | `'user'` |  |
| `ASSISTANT` | `'assistant'` |  |
| `TOOL` | `'tool'` |  |

### `RUN_EVENT_SCHEMA_VERSION`

*constant* · `libraries/rollout/src/rollout/contracts/events.py`

```python
RUN_EVENT_SCHEMA_VERSION = 1
```

### `RunEvent`

*class* · `libraries/rollout/src/rollout/contracts/events.py`

```python
class RunEvent(ContractModel)
```

One entry in a run's event stream.

| Field | Type | Default | Description |
|---|---|---|---|
| `run_id` | `str` | required |  |
| `seq` | `int` | required | Position in this run's event stream, gapless from 0; `run.created` is 0. |
| `type` | `RunEventType` | required |  |
| `schema_version` | `int` | `RUN_EVENT_SCHEMA_VERSION` |  |
| `recorded_at` | `datetime` | required | Exposed to code as `run.now()` for inputs. |
| `payload` | `JsonValue` | `None` | Type-specific; see docs/libraries/rollout/contracts/run-events.md. |

### `RunEventType`

*class* · `libraries/rollout/src/rollout/contracts/events.py`

```python
class RunEventType(StrEnum)
```

The closed catalog of run events.

| Member | Value | Description |
|---|---|---|
| `RUN_CREATED` | `'run.created'` |  |
| `RUN_COMPLETED` | `'run.completed'` |  |
| `RUN_FAILED` | `'run.failed'` |  |
| `RUN_CANCEL_REQUESTED` | `'run.cancel_requested'` |  |
| `RUN_CANCELLED` | `'run.cancelled'` |  |
| `OBSERVATION_RECORDED` | `'observation.recorded'` |  |
| `REWARD_ASSIGNED` | `'reward.assigned'` |  |
| `TRAINING_EXCLUDED` | `'training.excluded'` |  |
| `OUTPUT_EMITTED` | `'output.emitted'` |  |
| `EFFECT_REQUESTED` | `'effect.requested'` |  |
| `EFFECT_COMPLETED` | `'effect.completed'` |  |
| `TOOLS_RESOLVED` | `'tools.resolved'` |  |
| `SANDBOXES_ACQUIRED` | `'sandboxes.acquired'` |  |

### `RunFailureClass`

*class* · `libraries/rollout/src/rollout/contracts/events.py`

```python
class RunFailureClass(StrEnum)
```

Why a run failed (the `class` of a `run.failed` event).

| Member | Value | Description |
|---|---|---|
| `TASK_ERROR` | `'task_error'` |  |
| `INVALID_OBSERVATION` | `'invalid_observation'` |  |

### `SampleLink`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class SampleLink(ContractModel)
```

How a request follows from an earlier one of its session, as its program says: `type` is a label
(`compaction_attempt`, `compaction`, `subagent_call`, `subagent_return`, or any other), and `source` the earlier
request's `effect_id`. A recording endpoint keeps it with the turn; others ignore it.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `str` | required |  |
| `source` | `str` | required |  |

### `SampleRequest`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class SampleRequest(ContractModel)
```

A request for one reply. Sampling parameters are not here: they belong to the policy.

| Field | Type | Default | Description |
|---|---|---|---|
| `effect_id` | `str` | required | Required: the idempotency key. |
| `arguments_digest` | `str` | required |  |
| `session_id` | `str` | required |  |
| `context` | `ContextDelta` | required |  |
| `tools` | `FrozenSequence[ToolSpecification]` | `()` | The subset exposed this turn; endpoints use only the model-visible fields. |
| `max_output_tokens` | `int \| None` | `None` | Must not exceed the contract's `max_output_tokens`. |
| `tool_choice` | `ToolChoice \| None` | `None` |  |
| `links` | `FrozenSequence[SampleLink]` | `()` | How it follows from earlier requests of its session. |

### `SampleResult`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class SampleResult(ContractModel)
```

Nothing here identifies the policy, weights version or engine.

| Field | Type | Default | Description |
|---|---|---|---|
| `message` | `Message` | required |  |
| `finish_reason` | `FinishReason` | required |  |
| `usage` | `Usage` | required |  |

### `session_id`

*function* · `libraries/rollout/src/rollout/contracts/identifiers.py`

```python
def session_id(run_id: str, model_slot: str) -> str
```

`{run_id}/{model_slot}`: one recorded session per model slot per run.

### `SessionIdentity`

*class* · `libraries/rollout/src/rollout/contracts/identifiers.py`

```python
class SessionIdentity
```

The parts of a `session_id`: `{run_id}/{model_slot}`.

| Field | Type | Default | Description |
|---|---|---|---|
| `owner` | `str` | required |  |
| `model_slot` | `str` | required |  |

**Methods**

- `@classmethod def parse(cls, session_id: str) -> 'SessionIdentity'`

### `spec_hash`

*function* · `libraries/rollout/src/rollout/contracts/digests.py`

```python
def spec_hash(specification: ToolSpecification) -> str
```

Covers only the model-visible fields of a tool specification.

### `TERMINAL_EVENT_TYPES`

*constant* · `libraries/rollout/src/rollout/contracts/events.py`

```python
TERMINAL_EVENT_TYPES = frozenset({RunEventType.RUN_COMPLETED, RunEventType.RUN_FAILED, RunEventType.RUN_CANCELLED})
```

### `Text`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class Text(ContractModel)
```

Plain text.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `Literal['text']` | `'text'` |  |
| `text` | `str` | required |  |

### `ToolCall`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class ToolCall(ContractModel)
```

A request by the model to call a tool. Appears only in ASSISTANT messages.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `Literal['tool_call']` | `'tool_call'` |  |
| `call_id` | `str` | required | Unique within the context; the `ToolResultBlock` that answers the call repeats it. |
| `name` | `str` | required |  |
| `arguments` | `Mapping[str, JsonValue]` | required | A JSON object. |

### `ToolChoice`

*type alias* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
type ToolChoice = ToolChoiceMode | NamedToolChoice
```

### `ToolChoiceMode`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class ToolChoiceMode(StrEnum)
```

Whether the model may, must not, or must call a tool.

| Member | Value | Description |
|---|---|---|
| `AUTO` | `'auto'` |  |
| `NONE` | `'none'` |  |
| `REQUIRED` | `'required'` |  |

### `ToolResult`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class ToolResult(ContractModel)
```

What a tool produces. Platform failures are exceptions, not results.

| Field | Type | Default | Description |
|---|---|---|---|
| `content` | `FrozenSequence[ResultBlock]` | `()` |  |
| `structured` | `JsonValue` | `None` | Optional: the result as JSON, for code that reads it. |
| `is_error` | `bool` | `False` | A tool-level error the model should see and reason about (non-zero exit, file not found). |
| `truncated` | `bool` | `False` | `content` is only part of what the tool produced. |

### `ToolResultBlock`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class ToolResultBlock(ContractModel)
```

A tool's result as it appears in a TOOL message, answering the `ToolCall` with the same `call_id`.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `Literal['tool_result']` | `'tool_result'` |  |
| `call_id` | `str` | required |  |
| `result` | `ToolResult` | required |  |

### `ToolSpecification`

*class* · `libraries/rollout/src/rollout/contracts/content.py`

```python
class ToolSpecification(ContractModel)
```

What the model sees about a tool, plus an extension that is never model-visible.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `description` | `str` | `''` |  |
| `input_schema` | `Mapping[str, JsonValue]` | `Field(default_factory=lambda: {'type': 'object', 'properties': {}})` | JSON Schema 2020-12 with `type: object`. |
| `retry_class` | `RetryClass` | `RetryClass.UNKNOWN` |  |

**Methods**

- `def model_visible(self) -> dict[str, JsonValue]` — The fields a model sees and the spec hash covers.

### `Usage`

*class* · `libraries/rollout/src/rollout/contracts/model_endpoint.py`

```python
class Usage(ContractModel)
```

Context use after a sample.

| Field | Type | Default | Description |
|---|---|---|---|
| `context_used` | `int` | required | Drives the agent's compaction decisions. |
| `context_limit` | `int` | required |  |
| `input_tokens` | `int \| None` | `None` | Every prompt token, those read from a cache among them. |
| `output_tokens` | `int \| None` | `None` | Every token the reply took, its thinking among them. |
| `cached_input_tokens` | `int \| None` | `None` | Of `input_tokens`, those a provider read from its prompt cache (billed at its cached-input price). |
| `thinking_tokens` | `int \| None` | `None` | Of `output_tokens`, those spent thinking. |

## `rollout.environment`

What a run trains on and an eval measures: rows, starts, eval data, what results say.

### `binding_for`

*function* · `libraries/rollout/src/rollout/environment.py`

```python
def binding_for(environment: Environment, channel: str, tools: Mapping[str, ToolBinding] | None = None, pools: Mapping[str, PoolBinding] | None = None, slots: Mapping[str, str] | None = None) -> RunBinding
```

How an environment's runs are served: each model slot of its program from the channel `slots` names for it,
else from `channel`; each of its imports from the tool set of its own name, or where `tools` says; and each kind of
sandbox from the pool of its own name, or where `pools` says. (A program says which slots, imports and sandboxes it
has once it is given a row: the environment's first.)

### `Description`

*class* · `libraries/rollout/src/rollout/environment.py`

```python
class Description
```

What an environment's results say, for whatever shows or compares them; and how much an episode samples, for
estimating a run's spend (`turns`, `samples_per_turn`, `prompt_tokens`).

| Field | Type | Default | Description |
|---|---|---|---|
| `rewards` | `tuple[float \| None, float \| None]` | `(0.0, 1.0)` | The range an episode's reward falls in (None: no bound on that side). |
| `solved` | `bool` | `True` | Whether its results say `solved`. |
| `saturated` | `bool` | `False` | Whether its results say `saturated` (nothing was left to earn). |
| `duration` | `str \| None` | `None` | What a result's `duration` counts (`turns`, `minutes of game time`); None: its results say no duration. |
| `observations` | `str \| None` | `None` | How its observations are shown (`minecraft`, say); None: as text. |
| `turns` | `float \| None` | `None` | The turns an episode plays, at most, on average (its turn budgets, say); None: not said. |
| `samples_per_turn` | `float` | `1.0` | The samples a turn takes, on average: one for each model slot that samples in it (each agent of a team). |
| `prompt_tokens` | `int \| None` | `None` | A sample's prompt tokens, on average; None: not said. |

**Methods**

- `def to_json(self) -> dict[str, JsonValue]`

### `drawn`

*function* · `libraries/rollout/src/rollout/environment.py`

```python
def drawn(environment: Environment, *, seeds: Sequence[int], rows: Sequence[str] | None = None) -> list[Start]
```

A start of each row (of `rows`, by key; else every row) for each seed, drawn with `random.Random(seed)`: eval
data derived from rows and seeds. Raises `ValueError` for a row the environment lacks, or no seeds.

### `Environment`

*class* · `libraries/rollout/src/rollout/environment.py`

```python
class Environment(Protocol)
```

**Methods**

- `@property def program(self) -> ProgramReference` — What a run executes; its parameters are a start.
- `@property def version(self) -> str` — Changed whenever its rows, starts, eval data or scoring change: runs and suites record it.
- `@property def description(self) -> Description`
- `def rows(self) -> Sequence[Row]` — Every situation, easiest first.
- `def start(self, row: Row, rng: random.Random) -> JsonValue` — The parameters of one start of `row` (a seed drawn with `rng`, say): what every run of a group is given.
- `def evals(self) -> Mapping[str, Sequence[Start]]` — Its eval data: named lists of starts, never drawn for training. Each is frozen as a suite of its name the
  first time it is played (`rollout_train.evals`). `drawn` derives one from rows and seeds.

### `first_program`

*function* · `libraries/rollout/src/rollout/environment.py`

```python
def first_program(environment: Environment) -> ProgramReference
```

The environment's program given its first row's start (seed 0): what says which slots, imports and sandboxes
its runs have.

### `held_out`

*function* · `libraries/rollout/src/rollout/environment.py`

```python
def held_out(environment: Environment) -> frozenset[str]
```

The keys of every one of the environment's eval starts (`start_key`): what training never draws.

### `Row`

*class* · `libraries/rollout/src/rollout/environment.py`

```python
class Row
```

| Field | Type | Default | Description |
|---|---|---|---|
| `key` | `str` | required | Its name among the environment's rows. |
| `title` | `str` | required | What it is, for people. |
| `parameters` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` |  |
| `counts_for` | `tuple[str, ...]` | `()` | The keys of other rows that a group of this one is evidence about too: the same situation with more help, say. What it teaches about this row it teaches about them. |

### `Start`

*class* · `libraries/rollout/src/rollout/environment.py`

```python
class Start
```

One start of a row, drawn with a seed of its own: what an eval plays.

| Field | Type | Default | Description |
|---|---|---|---|
| `task` | `str` | required | The row's key. |
| `title` | `str` | required |  |
| `seed` | `int` | required |  |
| `parameters` | `JsonValue` | required | What every episode of it is given: the row's start, drawn with `seed`. |

### `start_key`

*function* · `libraries/rollout/src/rollout/environment.py`

```python
def start_key(parameters: JsonValue) -> str
```

A start's parameters as canonical JSON: two starts are the same start when their keys are equal.

### `train_start`

*function* · `libraries/rollout/src/rollout/environment.py`

```python
def train_start(environment: Environment, row: Row, rng: random.Random, held: Collection[str]) -> JsonValue
```

A start of `row` for training, drawn with `rng`, and drawn again while it is an eval start (its key in `held`,
`held_out`). Raises `ValueError` when `DRAWS` draws in a row are.

## `rollout.curriculum`

Which row to train on next, and gates on evals.

### `Curriculum`

*class* · `libraries/rollout/src/rollout/curriculum.py`

```python
class Curriculum
```

| Field | Type | Default | Description |
|---|---|---|---|
| `rows` | `Sequence[Row]` | required |  |
| `rng` | `random.Random` | `field(default_factory=random.Random)` |  |
| `start` | `int` | `3` | This many rows are unlocked from the beginning. |
| `reach` | `int` | `4` | Rows unlocked past the hardest one solved. |
| `smoothing` | `float` | `0.5` | Weight of the newest group in the moving averages. |
| `floor` | `float` | `0.05` |  |
| `records` | `dict[str, Record]` | `field(default_factory=dict[str, Record])` |  |
| `evaluations` | `dict[tuple[str, str \| None], tuple[str \| None, list[GroupResult]]]` | `field(default_factory=dict[tuple[str, str \| None], tuple[str \| None, list[GroupResult]]])` | The newest eval of each suite's entry, by the suite's name and the entry's environment: the checkpoint that played it (None: the base model), and how it did at each start of that entry. |

**Methods**

- `def unlocked(self) -> list[Row]`
- `def sample(self, pending: Collection[str] = (), rng: random.Random | None = None) -> Row` — The next row. `pending` names rows whose latest group has not been recorded yet: choosing one again would
  be choosing on what was known before it, so the others come first.
- `def recorded(self, line: GroupResult) -> None` — Take a group's result into account: the row of its title, or failing that of its key (a key that is a
  place in an environment changes when rows are added). A curriculum is the fold of a run's results.
- `def weight(self, row: Row) -> float`
- `def update(self, row: Row, rewards: Sequence[float], solved: Sequence[bool]) -> None` — Record a group of episodes of `row`: each one's reward and whether it solved the row.
- `def failed(self, row: Row) -> None` — Record a group of `row` none of whose episodes completed. After `FAILED_GROUPS` of them in a row it is no
  longer untried, and it taught nothing.
- `def evaluated(self, suite: str, checkpoint: str | None, results: Sequence[GroupResult], entry: str | None = None) -> None` — Take an eval's entry into account: `suite` played by `checkpoint` (None: the base model), one result per
  start of its entry of the environment `entry` (`module:name`; None where the eval does not say). The generic
  curriculum keeps the newest of each suite's entry and decides nothing from it; one that gates on evals
  overrides this.
- `def record(self, row: Row) -> Record`

### `curriculum_of`

*function* · `libraries/rollout/src/rollout/curriculum.py`

```python
def curriculum_of(environment: Environment) -> Curriculum
```

A curriculum that has recorded nothing: the environment's own (`environment.curriculum()`, if it has one), else
the generic one over its rows.

### `GroupResult`

*class* · `libraries/rollout/src/rollout/curriculum.py`

```python
class GroupResult(Protocol)
```

A group's result, as a curriculum reads it (`rollout_train.record.Result` is one).

**Methods**

- `@property def task(self) -> str` — The row's key.
- `@property def title(self) -> str`
- `@property def rewards(self) -> Sequence[float]` — Of the episodes that completed, as is `solved`.
- `@property def solved(self) -> Sequence[bool]`

### `solved_share`

*function* · `libraries/rollout/src/rollout/curriculum.py`

```python
def solved_share(results: Sequence[GroupResult]) -> float
```

The share of the episodes of `results` that solved their row (0 when there are none).

## `rollout.local`

The runner in this process.

### `EndpointFactory`

*type alias* · `libraries/rollout/src/rollout/local/runner.py`

```python
type EndpointFactory = Callable[[DirectModel], ModelEndpoint]
```

Creates the endpoint for a direct model binding; registered with the runner by provider name.

### `LocalRunContext`

*class* · `libraries/rollout/src/rollout/local/context.py`

```python
class LocalRunContext
```

Implements `RunContext` and `Effects` in process.

**Methods**

- `def __init__(self, run_id: str, endpoints: Mapping[str, ModelEndpoint], *, context_hints: ContextHints | None = None, tool_sets: Mapping[str, ToolSet] | None = None, blobs: Blobs | None = None, on_event: Callable[[RunEvent], None] | None = None) -> None`
- `@property def run_id(self) -> str`
- `@property def turn(self) -> int`
- `@property def history(self) -> History`
- `@property def models(self) -> Mapping[str, Model]`
- `@property def model(self) -> Model`
- `@property def tools(self) -> Tools`
- `def sandbox(self, name: str) -> Sandbox`
- `@property def blobs(self) -> Blobs | None`
- `@property def random(self) -> random.Random`
- `@property def context_hints(self) -> ContextHints`
- `def now(self) -> datetime`
- `def reward(self, value: float, *, slot: str = 'policy', key: str = 'default') -> None`
- `def exclude_from_training(self, reason: str) -> None`
- `async def gather[T](self, *awaitables: Awaitable[T]) -> list[T]`
- `async def emit(self, kind: str, payload: JsonValue) -> None` — Output of the run, such as its result. Recorded as an `output.emit` effect.
- `def record(self, observation: Observation, *, reply: Message | None = None) -> None`
- `async def acquire_sandboxes(self, specs: Mapping[str, SandboxSpec], pools: Mapping[str, Pool], lease: str) -> None` — Acquire each declared sandbox from the pool of its kind, under `lease` and its name, giving a harness
  inside it its slots' model addresses; then record them.
- `async def release_sandboxes(self) -> None` — Release every sandbox the run acquired, or began to: when the program has ended.
- `async def perform[T](self, kind: EffectKind, arguments: JsonValue, execute: Callable[[str, str], Awaitable[T]], *, completion: Callable[[T], JsonValue]) -> T`
- `def record_event(self, event_type: RunEventType, payload: JsonValue) -> RunEvent`

### `LocalRunHandle`

*class* · `libraries/rollout/src/rollout/local/runner.py`

```python
class LocalRunHandle
```

A run started by a `LocalRunner`. Its context is available for inspection in tests and tools.

**Methods**

- `def __init__(self, run_id: str, specification: RunSpecification, lease: str | None = None) -> None`
- `@property def run_id(self) -> str`
- `@property def done(self) -> bool`
- `@property def outcome(self) -> RunOutcome | None`
- `async def result(self) -> RunOutcome`
- `async def events(self, *, from_seq: int = 0) -> AsyncIterator[RunEvent]`
- `def recorded_events(self) -> list[RunEvent]` — Every event recorded so far.
- `@property def task(self) -> asyncio.Task[None] | None`
- `def attach(self, task: asyncio.Task[None]) -> None`
- `def finish(self, outcome: RunOutcome) -> None`
- `def notify(self, event: RunEvent | None = None) -> None` — Wake event streams: a new event was recorded, or the run ended.

### `LocalRunner`

*class* · `libraries/rollout/src/rollout/local/runner.py`

```python
class LocalRunner
```

Implements `Runner` in process. Direct model bindings are served by endpoint factories registered by provider
name.

**Methods**

- `def __init__(self, *, providers: Mapping[str, EndpointFactory] | None = None, tool_sets: Mapping[str, ToolSet] | None = None, blobs: Blobs | None = None, gateway: RecordedEndpoints | None = None, hooks: Sequence[RunHooks] = (), pools: Mapping[str, Pool] | None = None) -> None` — `gateway` serves recorded model bindings (trainable channels: a gateway's endpoints); direct bindings use
  `providers`. `pools`
  are the sandbox pools a binding names as `local`. `hooks` watch every run: each event recorded and each model
  sample.
- `async def launch(self) -> None` — Nothing to start: runs execute on the caller's event loop.
- `async def close(self) -> None` — Nothing to release: nothing outlives the process.
- `def run(self, run_id: str) -> LocalRunHandle`
- `async def start(self, specification: RunSpecification, *, run_id: str | None = None, labels: Mapping[str, str] | None = None, lease: str | None = None) -> LocalRunHandle`
- `async def cancel(self, run_id: str, *, reason: str) -> None`

### `RewardAssignment`

*class* · `libraries/rollout/src/rollout/local/context.py`

```python
class RewardAssignment
```

| Field | Type | Default | Description |
|---|---|---|---|
| `slot` | `str` | required |  |
| `value` | `float` | required |  |
| `key` | `str` | required |  |

## `rollout.testing`

Test doubles: a scripted model endpoint and helpers.

### `events_of` {#rollouttestingevents_of}

*function* · `libraries/rollout/src/rollout/testing.py`

```python
def events_of(run: LocalRunContext, event_type: RunEventType) -> list[RunEvent]
```

The run's events of one type, in order.

### `FakeSandbox`

*class* · `libraries/rollout/src/rollout/testing.py`

```python
class FakeSandbox
```

One of `FakeSandboxes`: what it was made from and given, the process it runs, and every operation asked of
it.

| Field | Type | Default | Description |
|---|---|---|---|
| `handle` | `str` | required |  |
| `spec` | `SandboxSpec` | required |  |
| `environment` | `Mapping[str, str]` | required |  |
| `process` | `Process \| None` | `None` | What its spec says to run, with the lease's environment added to the process's own. |
| `written` | `int` | `0` | Bytes written to its scratch directory. |
| `calls` | `list[tuple[str, Mapping[str, JsonValue]]]` | `field(default_factory=list[tuple[str, Mapping[str, JsonValue]]])` |  |

### `FakeSandboxes`

*class* · `libraries/rollout/src/rollout/testing.py`

```python
class FakeSandboxes
```

A sandbox `Provider` whose sandboxes are only records, honouring what their specs allow: at most `size` of
them, of `kind`. A spec's process is "launched" with the lease's environment added to its own, and the runner
reaches it at the address `process`. Its operations are `describe` (the sandbox's handle, parameters, environment
and process), `write` (`path`, `bytes`: only within its scratch directory and its size, never on a mount), `fetch`
(`host`: only a host its network allows), and any in `operations`; it keeps every sandbox it made and deleted.
Lease them out with `SandboxPool(FakeSandboxes())`.

**Methods**

- `def __init__(self, kind: str = 'fake', size: int = 4, operations: Mapping[str, Operation] | None = None) -> None`
- `def operations(self) -> Sequence[ToolSpecification]`
- `async def create(self, handle: str, spec: SandboxSpec, environment: Mapping[str, str]) -> Reach`
- `async def delete(self, handle: str) -> None`
- `async def held(self) -> Sequence[str]`
- `async def call(self, handle: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str) -> ToolResult`

### `LedgerEndpoint`

*class* · `libraries/rollout/src/rollout/testing.py`

```python
class LedgerEndpoint
```

Wraps a model endpoint and appends every sample's `effect_id` to a file: to count calls across processes.

**Methods**

- `def __init__(self, inner: ModelEndpoint, ledger: Path) -> None`
- `def describe(self, session_id: str) -> CapabilityContract`
- `def address(self, session_id: str) -> ModelAddress`
- `async def sample(self, request: SampleRequest) -> SampleResult`
- `async def cancel(self, effect_id: str) -> None`

### `local_run`

*function* · `libraries/rollout/src/rollout/testing.py`

```python
def local_run(task: Task, replies: Iterable[ScriptedReply] = ()) -> tuple[LocalRunContext, ScriptedModelEndpoint]
```

A local run context for `task` whose model slots all reply from one script.

### `payload`

*function* · `libraries/rollout/src/rollout/testing.py`

```python
def payload(event: RunEvent) -> dict[str, JsonValue]
```

An event's payload as a JSON object.

### `ScriptedModelEndpoint`

*class* · `libraries/rollout/src/rollout/testing.py`

```python
class ScriptedModelEndpoint
```

Replies with the scripted entries in order and records every request and cancellation.

**Methods**

- `def __init__(self, replies: Iterable[ScriptedReply], *, contract: CapabilityContract | None = None) -> None`
- `def describe(self, session_id: str) -> CapabilityContract`
- `async def sample(self, request: SampleRequest) -> SampleResult`
- `async def cancel(self, effect_id: str) -> None`

### `ScriptedReply`

*type alias* · `libraries/rollout/src/rollout/testing.py`

```python
type ScriptedReply = Message | str | Callable[[SampleRequest], Message | Awaitable[Message]]
```

A reply, its text, or a function of the request (which may await, e.g. to hold a sample open).

### `tool_call_reply`

*function* · `libraries/rollout/src/rollout/testing.py`

```python
def tool_call_reply(*calls: ToolCall, text: str = '') -> Message
```

An assistant reply that makes tool calls.

### `until`

*function* · `libraries/rollout/src/rollout/testing.py`

```python
async def until(condition: Callable[[], object], seconds: float = 15.0, *, every: float = 0.01, message: str = 'not met in time') -> None
```

Wait until `condition()` is true, asking every `every` seconds; it may return an awaitable, which is awaited.
Raises `AssertionError` (with `message`) once `seconds` have passed.

## `rollout_train.rollouts`

Episodes a run asks for in the ledger, claimed and played by runners, and read back.

### `Episode`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
class Episode
```

| Field | Type | Default | Description |
|---|---|---|---|
| `run` | `str` | required | The training run (or other caller) that asked for it. |
| `group` | `int` | required |  |
| `number` | `int` | required | Its number in its group, from 1. |
| `run_id` | `str` | required | The program's run that played it. |
| `labels` | `Mapping[str, str]` | required |  |
| `outcome` | `Outcome` | required |  |
| `detail` | `str \| None` | `None` |  |
| `info` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | What the program reported as its result (`run.emit("result", {...})`). `solved`, `saturated` and `duration` read the three entries training knows about. |
| `excluded` | `str \| None` | `None` | Why the program asked for the run to be left out of training, if it did. |
| `trajectories` | `Mapping[str, Trajectory]` | `field(default_factory=dict[str, Trajectory])` |  |

**Methods**

- `@property def reward(self) -> float` — The mean of the trained slots' rewards (a team that is rewarded together has one reward).
- `@property def trainable(self) -> bool`
- `@property def solved(self) -> bool` — Whether the program said its task was solved (`info["solved"]`).
- `@property def saturated(self) -> bool` — Whether the program said nothing was left to earn (`info["saturated"]`).
- `@property def duration(self) -> float | None` — How long the program said it took, in the task's own units (`info["duration"]`), if it said.

### `EpisodeRunner`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/scheduler.py`

```python
class EpisodeRunner
```

Claims the episodes runs ask for in `ledger` and plays them on `runner`, at most `places` at once: those of the
runs it can serve (whose models its recorder samples, whose imports are among `imports` and whose local
pools are among `pools`), and of `runs` only, if given. An episode is claimed only while the pools of its
sandboxes have room for them, and its run's sandboxes are leased under its claim. `guard` is called before
claiming and raises to wait (a machine short of memory, say). With `presence`, it beats every `beating` seconds,
with what `about` says of its machine besides its places, how many it plays and how full its pools are, and a
claim holds only while its runner beats. Over a runner whose runs survive it (`resumes`), closing leaves its runs
to be resumed, and starting again adopts them (`prepare`).

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `ledger` | `Ledger` | required |  |
| `runner` | `Runner` | required |  |
| `recorder` | `Recorded` | required |  |
| `blobs` | `Blobs` | required |  |
| `places` | `int` | required |  |
| `imports` | `Collection[str]` | `()` |  |
| `runs` | `Collection[str] \| None` | `None` |  |
| `hooks` | `Sequence[Hooks]` | `()` |  |
| `guard` | `Callable[[], None] \| None` | `None` |  |
| `presence` | `Presence \| None` | `None` |  |
| `about` | `Callable[[], Mapping[str, JsonValue]] \| None` | `None` | What the runner says of its machine in each beat (called in a thread: it may measure). |
| `pools` | `Mapping[str, Pool]` | `field(default_factory=dict[str, Pool])` | The sandbox pools its runner has, by the name a binding gives as `local`. |
| `every` | `float` | `0.5` | Seconds between looks for work while nothing ends. |
| `beating` | `float` | `15.0` | Seconds between beats. |

**Methods**

- `@property def resumes(self) -> bool` — Whether its runner's runs survive it (`Runner.resumes`).
- `async def prepare(self) -> None` — Take its fence, beat, and adopt what it finds of its runs: over a runner whose runs survive it, call this
  before launching the runner, so that the runs it recovers find their claims holding. `serve` calls it if it
  has not been.
- `async def serve(self) -> None` — Claim and play episodes until cancelled; what is playing then is cut short and noted (over a runner whose
  runs survive it, left to be resumed).
- `async def beat(self) -> None` — Beat now, beside the beats every `beating` seconds: after what it says of itself changed (a channel serves
  a new checkpoint, say), so that whoever reads the beats does not wait for the next.
- `async def open(self) -> list[Open]` — The episodes nobody plays now, of the runs this runner serves that are not paused, oldest group first. Which
  of them are paused is noted, and said at once in a beat when it changed.

### `episodes_of`

*function* · `libraries/rollout-train/src/rollout_train/rollouts/scheduler.py`

```python
async def episodes_of(ledger: Ledger, blobs: Blobs, run: str, group: int, count: int, *, every: float = 0.5) -> list[Episode]
```

A group's episodes once all `count` have ended, waiting for them.

### `events_of` {#rollout_trainrolloutsevents_of}

*function* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
async def events_of(record: Record, blobs: Blobs) -> list[RunEvent]
```

The events of the run a record names, as its runner recorded them.

### `Hooks`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/scheduler.py`

```python
class Hooks(Protocol)
```

Watch a runner (episodes as they start and end) or a run (its results and steps).

**Methods**

- `def on_note(self, event: Mapping[str, JsonValue]) -> None` — `event["kind"]` is `started` or `ended` (an episode, by a runner: an adopted one is started again), or the
  run's `result`, `step` or `published`.

### `loaded` {#rollout_trainrolloutsloaded}

*function* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
async def loaded(record: Record, blobs: Blobs) -> Episode
```

The episode a record names, with its trajectories read back from `blobs`.

### `Outcome`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
class Outcome(StrEnum)
```

| Member | Value | Description |
|---|---|---|
| `COMPLETED` | `'completed'` |  |
| `FAILED` | `'failed'` | The program raised: its `detail` says what. |
| `CANCELLED` | `'cancelled'` |  |

### `Plan` {#rollout_trainrolloutsplan}

*class* · `libraries/rollout-train/src/rollout_train/rollouts/scheduler.py`

```python
class Plan
```

How a run's episodes are played: its program (each group's start is its row) and its binding.

| Field | Type | Default | Description |
|---|---|---|---|
| `program` | `ProgramReference` | required |  |
| `binding` | `RunBinding` | required |  |

**Methods**

- `def to_json(self) -> dict[str, JsonValue]`
- `@classmethod def from_json(cls, data: Mapping[str, Any]) -> 'Plan'`

### `plan`

*function* · `libraries/rollout-train/src/rollout_train/rollouts/scheduler.py`

```python
async def plan(ledger: Ledger, run: str, played: Plan, fence: Fence) -> None
```

Say how a run's episodes are played, from now on (each start of a run may say it anew).

### `playing`

*function* · `libraries/rollout-train/src/rollout_train/rollouts/scheduler.py`

```python
async def playing(runner: EpisodeRunner) -> AsyncGenerator[None]
```

`async with playing(runner):` — the runner serves while the block runs.

### `Record`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
class Record
```

An episode as it is logged and sent: everything but its trajectories, and where those and its events are kept.

| Field | Type | Default | Description |
|---|---|---|---|
| `episode` | `Episode` | required | With no segments in its trajectories: their rewards only. |
| `trajectories` | `BlobReference \| None` | `None` |  |
| `events` | `BlobReference \| None` | `None` |  |
| `sampled` | `Mapping[str, int]` | `field(default_factory=dict[str, int])` | Tokens the policy sampled, by model slot. |

**Methods**

- `def to_json(self) -> dict[str, Any]`
- `@classmethod def from_json(cls, data: Mapping[str, Any]) -> 'Record'`

### `Recorded`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/scheduler.py`

```python
class Recorded(Protocol)
```

What a runner needs of what records its runs' samples (`rollout_train.gateway.GatewayEndpoints`).

**Methods**

- `def admit(self, run_id: str, attempt: Attempt) -> None` — Say which attempt a run plays (its episode's fence), before it starts or is adopted.
- `async def reaches(self, run: str, binding: RunBinding) -> bool` — Whether every recorded model of a run's binding can be sampled now.
- `async def sessions(self, run: str, run_id: str) -> dict[str, list[Segment]]` — What each model slot of a run recorded, by slot.
- `def forget(self, run_id: str) -> None`

### `stored` {#rollout_trainrolloutsstored}

*function* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
async def stored(episode: Episode, events: Sequence[RunEvent], blobs: Blobs) -> Record
```

Keep an episode's trajectories and its run's events in `blobs`; returns the record that names them.

### `Trajectory`

*class* · `libraries/rollout-train/src/rollout_train/rollouts/episodes.py`

```python
class Trajectory
```

What one model slot's rollout leaves to train on: its segments, and its rewards.

| Field | Type | Default | Description |
|---|---|---|---|
| `segments` | `list[Segment]` | required |  |
| `rewards` | `Mapping[str, float]` | required | By key; a program that assigns one reward uses the key `default`. |
| `trained` | `bool` | `True` | Whether its slot is trained: false for a judge's or a fixed opponent's, whose turns are kept and never trained on (`Segment.trained`), and whose rewards, if any, are not the episode's. |

**Methods**

- `@property def reward(self) -> float`

## `rollout_train.sandboxes`

Sandboxes' leases beside the ledger, each ending with its episode's claim.

### `admits`

*function* · `libraries/rollout-train/src/rollout_train/sandboxes.py`

```python
def admits(ledger: Ledger, presence: Presence | None) -> Callable[[str], Awaitable[bool]]
```

For a pool beside a ledger (`SandboxPool(admits=...)`): whether a key may hold a lease now. A key whose run the
ledger knows may while its claim holds; any other key may.

### `ended`

*function* · `libraries/rollout-train/src/rollout_train/sandboxes.py`

```python
async def ended(leases: list[Lease], ledger: Ledger, presence: Presence | None) -> Callable[[Lease], bool]
```

Which of `leases` have ended, as the ledger says now: those whose run it knows and whose claim does not hold.

### `ending`

*function* · `libraries/rollout-train/src/rollout_train/sandboxes.py`

```python
async def ending(keys: Collection[str], ledger: Ledger, presence: Presence | None) -> set[str]
```

Of the keys of leases found ended, those that may be released now. Each claim is read again: one that holds
again (adopted meanwhile) keeps its lease. One that is still its episode's latest attempt, and neither cut short nor
recorded, is ended in the ledger first: its episode's fence is taken, and the attempt noted cut short under it. One
whose note is refused (another took the fence meanwhile) is left for the next look.

### `FileLeases`

*class* · `libraries/rollout-train/src/rollout_train/sandboxes.py`

```python
class FileLeases
```

`Leases` in `sandboxes.json` in a ledger's directory, under the lock the ledger's files are written under.

**Methods**

- `def __init__(self, directory: Path) -> None`
- `async def get(self, key: str) -> Lease | None`
- `async def put(self, lease: Lease) -> None`
- `async def delete(self, key: str) -> None`
- `async def all(self) -> list[Lease]`

### `keep`

*function* · `libraries/rollout-train/src/rollout_train/sandboxes.py`

```python
async def keep(pool: SandboxPool, ledger: Ledger, presence: Presence | None, *, beat_as: str | None = None, every: float = 15.0) -> None
```

Sweep the pool every `every` seconds, until cancelled or another process takes the pool's fence, releasing a
lease once its claim was found lapsed at two looks running and again just before (`ending`); with `beat_as`, beat
under that name too.

### `leases_of`

*function* · `libraries/rollout-train/src/rollout_train/sandboxes.py`

```python
def leases_of(ledger: Ledger) -> Leases | None
```

The leases beside a ledger: a file beside a ledger of files, a table in a database ledger's database.

### `pool_scope`

*function* · `libraries/rollout-train/src/rollout_train/sandboxes.py`

```python
def pool_scope(name: str) -> str
```

The scope whose fence a pool's keeper holds while it sweeps (`pools/NAME`).

### `sweep`

*function* · `libraries/rollout-train/src/rollout_train/sandboxes.py`

```python
async def sweep(pool: SandboxPool, ledger: Ledger, presence: Presence | None, *, lapsed: Collection[str] | None = None) -> tuple[list[str], set[str]]
```

Release the pool's leases whose claims have ended (given `lapsed`, only those whose claims were found ended the
look before too: the keys it holds), ending the claims in the ledger first (`ending`), and delete what no lease
names. Returns the keys released, and the keys of the leases whose claims were found ended now.

## `rollout_train`

The training loop, the group algorithm, evals, and what they ask of a trainer.

### `Algorithm`

*class* · `libraries/rollout-train/src/rollout_train/algorithm.py`

```python
class Algorithm(Protocol)
```

What the training loop asks of an algorithm.

**Methods**

- `@property def group_size(self) -> int` — How many episodes of one start it compares.
- `def batch(self, group: Sequence[Episode], budget: Budget, rng: random.Random) -> Batch[Item]` — What to train on from a group's episodes (of every outcome), within what the trainer can afford.

### `algorithm_for`

*function* · `libraries/rollout-train/src/rollout_train/algorithm.py`

```python
def algorithm_for(objective: Objective, group_size: int | None = None) -> Grpo | Preferences | Distillations
```

The algorithm that makes the batch items an objective's family takes, comparing `group_size` episodes of one
start (none: 4, and 1 for a distillation, which compares nothing).

### `Batch`

*class* · `libraries/rollout-train/src/rollout_train/algorithm.py`

```python
class Batch[Each: Item]
```

What an algorithm makes of a group of episodes: items of one kind (`Each`).

| Field | Type | Default | Description |
|---|---|---|---|
| `items` | `Sequence[Each]` | `()` | What to train on. |
| `skipped` | `str \| None` | `None` | Why there is nothing to train on, if there is not. |
| `notes` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | What the algorithm wants logged with the group. |

### `Budget`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Budget
```

| Field | Type | Default | Description |
|---|---|---|---|
| `segment_tokens` | `int \| None` | `None` | The longest segment the trainer can train on (None: any). |
| `segments` | `int \| None` | `None` | How many segments a step can afford (None: any number), counting each of a pair's or an example's. |

### `Changeable`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Changeable(Protocol)
```

A trainer that takes some of its settings between steps (its learning rate, say): those that change neither
what its weights are nor what it can take (`Budget`).

**Methods**

- `@property def changeable(self) -> Mapping[str, JsonValue]` — The settings it takes between steps, by its name for each, with their values now: a component of its
  objective by its run setting's key (`objective.kl.coefficient`).
- `def change(self, settings: Mapping[str, JsonValue]) -> None` — Take these settings (some of `changeable`) from its next step on. Raises `ValueError` for one it does not
  take, or a value it cannot.

### `Checkpoint`

*class* · `libraries/rollout-train/src/rollout_train/checkpoints.py`

```python
class Checkpoint
```

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required |  |
| `weights` | `Manifest \| None` | required | None once it was released (`Checkpoints.thin`). |
| `parents` | `tuple[str, ...]` | `()` | What it was made from, by id: first the checkpoint it was trained from, then any others it learned from (the checkpoints that sampled a dataset's examples, say). Empty: from the base model. |
| `depth` | `int` | `1` | Steps from the base model along its first parents: its first parent's depth and one. |
| `base` | `str \| None` | `None` | What its weights build on: the model its line began from, by name (`Qwen/Qwen3.5-9B`, say); or, for an adapter trained over a full checkpoint, that checkpoint, by id. |
| `kind` | `str` | `'lora'` | What its weights are: `lora` (an adapter over its base) or `full` (all of a model's weights). |
| `run` | `str \| None` | `None` | The run that made it, by id. |
| `step` | `int \| None` | `None` | The run's step that made it (none for a checkpoint made outside a run's steps, such as by imitation). |
| `state` | `Manifest \| None` | `None` | What a trainer goes on from: the optimizer's state, say. While `state_complete` is false, only what was kept with the weights (the name a trainer gave what it holds, and what the step did). |
| `state_complete` | `bool` | `True` | Whether `state` is all the trainer left: false while the trainer is still keeping the rest (`Checkpoints.completed`), and for good if it never kept it. |
| `state_seconds` | `float \| None` | `None` | How long the trainer took to keep the rest of the state after the step returned, for a state completed so. |
| `batch` | `BlobReference \| None` | `None` | What it was trained on: the segments, each as its source (`RUN/GROUP/EPISODE/SLOT/INDEX`) and its advantage. |
| `metrics` | `Mapping[str, float]` | `field(default_factory=dict[str, float])` |  |
| `dataset` | `str \| None` | `None` | The dataset it was trained on, by id (`rollout_train.datasets`), if a supervised step on one made it: its parents after the first are then the checkpoints that sampled the dataset's examples. |
| `supervision` | `str \| None` | `None` | For a checkpoint a supervised step made: `teacher` if a teacher scored every segment it trained on with its top-k, else `importance` if every one was sampled with its exact tokens and behaviour logprobs, else `supervised` (`rollout_train.imitation.supervision_of`). |
| `made` | `float` | `0.0` | When, in seconds since the epoch. |
| `released` | `float \| None` | `None` | When its files were deleted (`Checkpoints.thin`), if they were: its weights and its trainer state are then None. Its record stays: where it came from, what it was trained on, and its metrics. |

**Methods**

- `@property def parent(self) -> str | None` — The checkpoint it was trained from, if any.

### `Checkpoints`

*class* · `libraries/rollout-train/src/rollout_train/checkpoints.py`

```python
class Checkpoints
```

Every checkpoint, in a ledger, and their files in a blob store.

**Methods**

- `def __init__(self, ledger: Ledger, blobs: Blobs) -> None`
- `async def all(self) -> list[Checkpoint]` — Every checkpoint, oldest first.
- `async def checkpoint(self, id: str) -> Checkpoint` — The checkpoint an id says.
- `async def under(self, checkpoint: Checkpoint) -> Checkpoint | None` — The full checkpoint whose weights `checkpoint` is served over: itself, if it is full; the full checkpoint it
  builds on, if it is an adapter over one; None for an adapter over a model. Raises `ValueError` if that
  checkpoint was released.
- `async def head(self, run: str) -> Checkpoint | None` — The newest checkpoint a run made, if it made one.
- `async def add(self, fence: Fence, id: str, *, weights: Path | Manifest, run: str | None, base: str | None = None, kind: str = 'lora', step: int | None = None, state: Path | Manifest | None = None, state_complete: bool = True, parents: Sequence[str] = (), batch: BlobReference | None = None, metrics: Mapping[str, float] | None = None, dataset: str | None = None, supervision: str | None = None) -> Checkpoint` — Append the checkpoint that names its files, under `fence` (the run's that makes it): `weights` and `state`
  are manifests of files in the blob store, or directories on this machine, whose files are kept first. A state
  the trainer is still keeping is added incomplete (`state_complete` false: what was kept with the weights), and
  completed once it is kept (`completed`). Its base is what its weights build on (`_base`): for an adapter over a
  full checkpoint, that checkpoint (by id); for a merge (full weights from an adapter), the `base` it names; else
  its first parent's base, or `base` for a checkpoint made from the base model. The append is what makes the
  checkpoint exist: a writer that dies before it has made nothing, and one that repeats it (the same id, decided
  before) gets the checkpoint that is there.
- `async def completed(self, fence: Fence, id: str, state: Manifest, *, seconds: float | None = None) -> Checkpoint` — Complete a checkpoint's trainer state, kept after it was added: `state` is the whole state (what was kept
  with the weights among it), and `seconds` how long keeping the rest took. Appended under `fence` (the run's
  that made it); a checkpoint completed before stays as it was.
- `async def thin(self, fence: Fence, run: str, retention: 'Retention', keep: Collection[str] = ()) -> list[str]` — Delete the files (weights and trainer state) of the checkpoints `run` made that `retention` does not keep,
  nor `keep` (what is served, what is bookmarked, what another run starts from), and return their ids. A
  release is appended to the ledger before its blobs are deleted, and a blob is deleted only if nothing still
  names it (a checkpoint that was not released, or what a bridge made of one), so this may be repeated after a
  crash at any point. Nor is a blob deleted that was put in the last `retention.grace` seconds: a checkpoint being
  added at the same moment, which found the blob stored and has not appended itself yet, names it next. A blob
  spared so is deleted by a later thinning, of this run or any other.
- `async def files(self, manifest: Manifest, directory: Path) -> Path` — A manifest's files under `directory`, read from the blob store if they are not there. The directory
  appears whole or not at all, so whatever looks for a file in it never finds half a checkpoint. A file this
  store lacks is read from the store of any run that has it (a checkpoint made by a run that kept its blobs
  elsewhere, or a merge of one), as each run's start says where its store is.

### `Colocated`

*class* · `libraries/rollout-train/src/rollout_train/colocated.py`

```python
class Colocated
```

A trainer that shares an accelerator with the engines of some channels: requests to them are held back and
the engines sleep while it steps. `guard` is called once they are asleep and raises if the step should not
start (too little memory, say).

**Methods**

- `def __init__(self, trainer: Trainer, channels: Sequence[Pausable], *, guard: Callable[[], None] | None = None) -> None`
- `@property def changeable(self) -> Mapping[str, JsonValue]` — The settings the trainer it wraps takes between steps (`rollout_train.trainer.Changeable`), if any.
- `def change(self, settings: Mapping[str, JsonValue]) -> None`
- `@property def holding(self) -> str | None` — What the trainer it wraps holds between steps (`rollout_train.trainer.Resident`), if it holds anything.
- `def close(self) -> None` — End what the trainer it wraps keeps running between steps, if it keeps anything.
- `async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step`

### `Dataset`

*class* · `libraries/rollout-train/src/rollout_train/datasets.py`

```python
class Dataset
```

A dataset's record (`DATASETS`): how its examples were chosen, what came of it, and where its manifest is.

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required | Sixteen random letters, like a checkpoint's. |
| `rule` | `str` | required | The episode rule, by name (`RULES`). |
| `runs` | `list[str]` | required | The runs its episodes are from, by id. |
| `turns` | `list[str]` | required | The turn filters, by name. |
| `cut` | `list[str]` | required | The kinds of guidance cut from its examples' prompts when they are made. |
| `manifest` | `BlobReference` | required | One JSON line per example, compressed (`manifest_of` reads it). |
| `blobs` | `Mapping[str, JsonValue]` | required | Where the manifest is kept, as any process opens it (`rollout_train.stores`). |
| `per_task` | `int \| None` | `None` | For `capped-per-task`: the most episodes of each task. |
| `counts` | `Mapping[str, int]` | `field(default_factory=dict[str, int])` | `episodes` the rule picked and the `groups` they are of; `turns_seen`, every turn of those episodes; and of its examples, `tasks`, `turns`, `sampled_tokens` and `context_tokens`. |
| `left_out` | `Mapping[str, int]` | `field(default_factory=dict[str, int])` | Turns of its episodes that are no examples, by why. |
| `checkpoints` | `list[str]` | `field(default_factory=list[str])` | The checkpoints that sampled its examples, by id, by depth (examples sampled by the base model name none). |
| `supervision` | `str` | `IMPORTANCE` | `teacher` if a teacher scored every example with its top-k (a dataset of examples), else `importance` if every example's turns were sampled with their exact tokens and behaviour logprobs, else `supervised`. |
| `kind` | `str` | `EXAMPLES` | What its lines are: `examples`, `pairs` or `labelled` examples. |
| `made` | `float` | `0.0` | When, in seconds since the epoch. |
| `by` | `str` | `''` | Who made it: `user@host`. |

### `dataset_of`

*function* · `libraries/rollout-train/src/rollout_train/datasets.py`

```python
async def dataset_of(ledger: Ledger, id: str) -> Dataset
```

The dataset an id says.

### `Distillations`

*class* · `libraries/rollout-train/src/rollout_train/algorithm.py`

```python
class Distillations
```

Distilled segments: every trained segment of a group's completed episodes, with the teacher's scores it carries
(for the distillation family). Nothing is compared, so a group is one episode by default (MOPD's one rollout per
prompt), and its reward is not read.

| Field | Type | Default | Description |
|---|---|---|---|
| `group_size` | `int` | `1` |  |
| `top_k` | `int` | `0` | The teacher's top tokens the objective reads at each sampled position. |
| `needs` | `tuple[str, ...]` | `('token_exact',)` | What every segment's turns must have been sampled with (`needs_of`). |

**Methods**

- `def batch(self, group: Sequence[Episode], budget: Budget, rng: random.Random) -> Batch[Distilled]` — The trained segments of the group's completed episodes, each with its teacher's scores; none, if one of
  them cannot be trained on (`unweighable`) or has no teacher's scores (`unscored`).

### `Distilled`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Distilled
```

A segment and a teacher's scores of its sampled tokens (for the distillation family), and its episode's
advantage (for a policy gradient with a distillation term; 0 for a distillation alone).

| Field | Type | Default | Description |
|---|---|---|---|
| `segment` | `Segment` | required |  |
| `scores` | `TeacherScores` | required | One for each sampled token, in the order of the segment's spans, and which teacher gave them. |
| `advantage` | `float` | `0.0` |  |
| `source` | `str` | `''` | `RUN/GROUP/EPISODE/SLOT/INDEX`. |

### `edit_suite`

*function* · `libraries/rollout-train/src/rollout_train/evals.py`

```python
async def edit_suite(ledger: Ledger, name: str, entries: Sequence[SuiteEntry], *, base: int | None = None) -> Suite
```

Make a suite's next version of these entries, and point its name to it. An entry whose starts are those of the
current version's entry of its environment keeps how they were chosen. `base` is the version the edit was made
from, by number: an edit of another than the one the name points to is refused (someone edited it meanwhile).
Raises `KeyError` for a suite there is not; `ValueError` for an edit that changes nothing, or what `make_suite`
refuses.

### `evaluate`

*function* · `libraries/rollout-train/src/rollout_train/evals.py`

```python
async def evaluate(checkpoints: Checkpoints, *, run: str, suite: Suite, subject: str | None, base: str | None, channel: str, directory: Path, publish: Publisher | None, environments: Mapping[str, Environment] | None = None, binding: Callable[[Environment], RunBinding] | None = None, parts: Callable[[int], Awaitable[str]] | None = None, episodes: int | None = None, started: Mapping[str, JsonValue] | None = None, asked_by: str = 'by hand', reshard: Callable[[Checkpoint, Fence], Awaitable[Manifest]] | None = None, hooks: Sequence[Hooks] = ()) -> dict[str, Any]
```

Play `suite` (one version) with `subject` (a checkpoint's id; None: the base model, named `base`) served on
`channel`, `episodes` episodes of each start (none: each entry's), as the run `run`. Returns how it went: `played`,
`solved`, `reward` and every start's `results` (a `Result` each), and each entry's (`entries`: its environment, the
run that played it, its scores, `entry_scores`, and its `results`).

`environments` are the entries' environments by `module:name` (any not given are imported: `environments_of`);
`binding` says how an environment's episodes are played (by default every slot from `channel`), and each entry's
binding takes its limits. A suite of several entries plays each in a run of its own: `parts` gives the run of an
entry, by its number from 1 (by default `RUN-NUMBER`). `publish` serves a checkpoint on the channel (a full one in
place of the engines' weights; for an adapter over a full checkpoint, the engines must already hold that
checkpoint's weights, as `rollout eval` sees to); None: the channel serves `subject` already (a training run's
newest checkpoint). `reshard` gives the files its engines load (made by a bridge: `rollout_train.bridges`);
`directory` holds its files on this machine. What the eval's channel serves is written down for each of its runs
(`rollout_train.serving`), so that runners anywhere play it on replicas that serve `subject` and no other
checkpoint. Raises `KeyError` for an entry's environment that does not load here.

### `Fence`

*class* · `libraries/rollout-train/src/rollout_train/ledger.py`

```python
class Fence
```

The right to write within a scope, until someone takes it again.

| Field | Type | Default | Description |
|---|---|---|---|
| `scope` | `str` | required |  |
| `number` | `int` | required |  |

### `Fenced`

*class* · `libraries/rollout-train/src/rollout_train/ledger.py`

```python
class Fenced(Exception)
```

A writer whose fence is no longer the newest tried to write: another has taken its place.

### `FileLedger`

*class* · `libraries/rollout-train/src/rollout_train/ledger.py`

```python
class FileLedger
```

A `Ledger` in a directory: a table is `<table>.jsonl`, one `{"key", "fence", "record"}` per line. Processes
on one machine may share it: every operation holds a lock on the directory, in a thread (the event loop never
waits on the lock).

An append is on disk (`fsync`) before it is acknowledged. A last line left unfinished (by a writer that died
mid-record, or a full disk) was never acknowledged: the next append removes it before writing (or ends it, where it
holds a whole record), so that nothing is glued to it. `fences.json` is replaced whole (written beside it and put on
disk, then renamed over it), so a crash while taking a fence leaves the fences as they were. Which keys a table has
is kept in memory, and read again only as far as its file grew since (other processes' appends), or whole if the
file was replaced.

**Methods**

- `def __init__(self, directory: Path) -> None`
- `async def take(self, scope: str) -> Fence`
- `async def append(self, table: str, key: str, record: JsonValue, fence: Fence) -> bool`
- `async def append_returning(self, table: str, key: str, record: JsonValue, fence: Fence) -> Appended` — `append`, saying what the table holds under `key` too.
- `async def read(self, table: str) -> dict[str, JsonValue]`
- `async def tables(self) -> list[str]`
- `async def read_all(self, *, leaving_out: str | None = None) -> dict[str, dict[str, JsonValue]]`
- `async def fences(self) -> dict[str, int]`

### `Files`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Files
```

A checkpoint's files on this machine: what a step starts from.

| Field | Type | Default | Description |
|---|---|---|---|
| `weights` | `Path` | required |  |
| `state` | `Path \| None` | `None` | What the trainer left for itself beside the weights (an optimizer's state, say), if it left any. |

### `Follower`

*class* · `libraries/rollout-train/src/rollout_train/following.py`

```python
class Follower
```

Keeps runs' channels serving what each run says they should, checking every `every` seconds, with each
checkpoint's files under `directory` while they are served.

Given `run` and `channels`, it serves those channels of that one run (by their names within it). Given `bindings`
(what it serves now, asked at each look) and `opened` (the channel for a binding new to it), it serves whatever
`bindings` says, and a binding it no longer says has its adapters removed. `replica` is this follower's index among
the replicas of its channels and how many there are. With `presence`, it beats as `name` every `beating` seconds,
and at once when a channel serves something new: what `about` says of the machine, and what each channel serves.

**Methods**

- `def __init__(self, name: str, checkpoints: Checkpoints, run: str | None, channels: Mapping[str, Channel], directory: Path, *, bindings: Callable[[], Awaitable[Collection[Binding]]] | None = None, opened: Callable[[str, str], Channel] | None = None, replica: tuple[int, int] = (0, 1), presence: Presence | None = None, about: Callable[[], Mapping[str, JsonValue]] | None = None, every: float = 2.0, beating: float = 15.0) -> None`
- `@property def channels(self) -> dict[str, Channel]` — The channels it serves, by `RUN/CHANNEL`.
- `async def serve(self) -> None` — Follow until cancelled.
- `async def follow(self) -> bool` — Give every channel what its run says it should serve, if it serves something older; whether any changed.
- `async def beat(self) -> None`
- `def served(self) -> list[JsonValue]` — What each channel serves, how fast since the last call, and each of its engines: its address (where it is a
  server elsewhere), what it serves, and every adapter it holds for any channel.

### `group_advantages`

*function* · `libraries/rollout-train/src/rollout_train/algorithm.py`

```python
def group_advantages(scores: Sequence[float]) -> list[float] | None
```

Each score minus the group's mean; None when all are equal (no signal): the `default` preset's advantages.

### `Grpo`

*class* · `libraries/rollout-train/src/rollout_train/algorithm.py`

```python
class Grpo
```

Weighted segments, each with its episode's advantage (for the policy-gradient and likelihood families); with
`distills`, distilled segments, each with its teacher's scores and its episode's advantage (a policy gradient with
a distillation term).

| Field | Type | Default | Description |
|---|---|---|---|
| `group_size` | `int` | `4` |  |
| `advantage` | `Advantage` | `DEFAULT_ADVANTAGE` |  |
| `needs` | `tuple[str, ...]` | `tuple(_LACKING)` | What every segment's turns must have been sampled with (`needs_of`). |
| `distills` | `bool` | `False` | Whether the objective has a distillation term: every segment is kept, a group whose advantages are all zero (or that the filter skips) too, since the teacher's scores train on it all the same. |
| `top_k` | `int` | `0` | For `distills`: the teacher's top tokens the objective reads at each sampled position. |

**Methods**

- `def batch(self, group: Sequence[Episode], budget: Budget, rng: random.Random) -> Batch[Weighted | Distilled]` — The segments of the group's episodes that are fit to train on (completed, and not excluded), each with
  its episode's advantage; none, if one of them cannot be weighed (`unweighable`) or, with `distills`, has no
  teacher's scores (`unscored`).

### `Keeps`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Keeps(Protocol)
```

A trainer that can keep the full state of its steps in a blob store itself, after each step returns, so a step's
weights are kept and served while its state is still being kept (`rollout_lora`'s, with its processes kept between
steps). Its steps say which they keep so (`Step.keeping`).

**Methods**

- `def keep_in(self, blobs: Mapping[str, JsonValue]) -> None` — Keep its steps' full state from now on in the blob store at this location (`rollout_train.stores.opened`),
  rather than in `into/state` before a step returns.
- `async def kept(self, into: str) -> Manifest` — The files it kept of the full state of the step that wrote into `into` (by its name), by their paths within
  the state, once they are kept. Raises `StateLost` if they never will be.

### `Labelled`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Labelled
```

The segments of one episode, labelled desirable or not (KTO's unpaired examples).

| Field | Type | Default | Description |
|---|---|---|---|
| `side` | `tuple[Segment, ...]` | required |  |
| `desirable` | `bool` | required |  |
| `source` | `str` | `''` | `RUN/GROUP/EPISODE`. |

**Methods**

- `@property def segments(self) -> tuple[Segment, ...]`

### `Ledger`

*class* · `libraries/rollout-train/src/rollout_train/ledger.py`

```python
class Ledger(Protocol)
```

**Methods**

- `async def take(self, scope: str) -> Fence` — Take a scope's fence. Whoever held it can no longer write within the scope.
- `async def append(self, table: str, key: str, record: JsonValue, fence: Fence) -> bool` — Append a record under `key`, unless the table has that key: then nothing changes and False is returned.
  Raises `Fenced` if `fence` is not its scope's newest.
- `async def append_returning(self, table: str, key: str, record: JsonValue, fence: Fence) -> Appended` — Append as `append` does, and say what the table holds under `key`: whether this call wrote `record`, and if
  it did not, the record appended first.
- `async def read(self, table: str) -> dict[str, JsonValue]` — A table's records by key, in the order they were appended: the order their appends took effect in, whichever
  scopes made them.
- `async def tables(self) -> list[str]` — The tables that have records, by name.
- `async def read_all(self, *, leaving_out: str | None = None) -> dict[str, dict[str, JsonValue]]` — Every table that has records, by name, each as `read` reads it; but the tables whose names hold
  `leaving_out`.
- `async def fences(self) -> dict[str, int]` — The newest fence of every scope that has been taken.

### `Made`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Made
```

What a step of a trainer elsewhere made (`Remote`), kept in the blob store: its metrics, the new weights, and the
state, whole or (`complete` false) only what was kept with the weights so far.

| Field | Type | Default | Description |
|---|---|---|---|
| `metrics` | `Mapping[str, float]` | required |  |
| `weights` | `Manifest` | required |  |
| `state` | `Manifest \| None` | `None` |  |
| `complete` | `bool` | `True` |  |

### `make_dataset`

*function* · `libraries/rollout-train/src/rollout_train/datasets.py`

```python
async def make_dataset(ledger: Ledger, rule: str, runs: Sequence[str], *, into: Blobs, at: Mapping[str, JsonValue], turns: Sequence[str] = (ALL,), cut: Sequence[str] = ('way',), per_task: int | None = None, by: str | None = None) -> Dataset
```

Make a dataset of the episodes `runs` (by id) completed: those `rule` picks (an episode rule, or a preference
rule's pairs or labelled examples), and of them the turns every filter of `turns` keeps. Its manifest is kept in
`into`, which is at `at` (as `rollout_train.stores.opened` reads it). Episodes are read one at a time, from where
each run's blobs are. Raises `ValueError` for a rule or filter that does not exist, or a dataset of no examples.

### `make_suite`

*function* · `libraries/rollout-train/src/rollout_train/evals.py`

```python
async def make_suite(ledger: Ledger, name: str, entries: Sequence[SuiteEntry]) -> Suite
```

Make version 1 of a suite of these entries (`suite_entry`). Raises `ValueError` for a name that is no name or is
taken (a suite is edited instead: `edit_suite`), no entries, or an environment in two.

### `Manifest`

*class* · `libraries/rollout-train/src/rollout_train/checkpoints.py`

```python
class Manifest
```

The files of a checkpoint, by their paths within it, each kept as a blob.

| Field | Type | Default | Description |
|---|---|---|---|
| `files` | `Mapping[str, BlobReference]` | required |  |
| `layout` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | How the weights are divided among the files, where they are divided: whoever wrote them says, so that a reader with the same division reads its own files and no others. |

### `Pair`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Pair
```

Two sides over a shared context (one start), the chosen preferred to the rejected: each the segments of one
episode (every turn the policy sampled in it), of which only the tokens the policy sampled count.

| Field | Type | Default | Description |
|---|---|---|---|
| `chosen` | `tuple[Segment, ...]` | required |  |
| `rejected` | `tuple[Segment, ...]` | required |  |
| `source` | `str` | `''` | Where the sides are from, for the record of what a step trained on: `RUN/GROUP/CHOSEN>REJECTED` (the episodes' numbers). |

**Methods**

- `@property def segments(self) -> tuple[Segment, ...]`

### `Preferences`

*class* · `libraries/rollout-train/src/rollout_train/algorithm.py`

```python
class Preferences
```

Pairs, a group's best episode preferred to its worst; or, `labelled`, examples, each episode above the group's
mean desirable and each below it undesirable (for the preference family).

| Field | Type | Default | Description |
|---|---|---|---|
| `group_size` | `int` | `4` |  |
| `labelled` | `bool` | `False` |  |

**Methods**

- `def batch(self, group: Sequence[Episode], budget: Budget, rng: random.Random) -> Batch[Pair | Labelled]` — A pair of the group's best and worst completed episodes (the first of each where several tie), or every
  completed episode not at the group's mean, labelled; none where every score is the same.

### `record_serving`

*function* · `libraries/rollout-train/src/rollout_train/serving.py`

```python
async def record_serving(ledger: Ledger, run: str, serving: Serving, fence: Fence) -> bool
```

Write down, under the run's fence, that its channel serves `serving` from now on; False if it was written
before (a loop started again serves what it served).

### `Remote`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Remote(Protocol)
```

A trainer on another machine, whose steps' files go through the blob store (`rollout_train.pods.RemoteTrainer`):
given its parent as the checkpoint, it says what it made as manifests (`Made`), so nothing is read to the loop's
machine. The rest of a step's state may be kept after `made` returns (`Made.complete` false): `state` waits for
it.

**Methods**

- `async def made(self, batch: Sequence[Item], *, seed: int, parent: Checkpoint | None, into: str) -> Made` — `Trainer.step`, from `parent`'s files in the blob store, making the checkpoint `into` (its id).
- `async def state(self, into: str) -> Manifest` — The whole state of the step that made `into`, once it is kept. Raises `StateLost` if it never will be.

### `Resident`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Resident(Protocol)
```

A trainer that keeps its policy and optimizer in memory between steps (`rollout_lora`'s, with its GPUs to
itself): a step from the checkpoint it made last goes on from them. Its steps still leave every file a later step
needs, so any trainer can take any step, unless it is told to leave its full state out of some (`rollout_lora`'s
`state_every`): a step from one of those needs the trainer that holds it, and fails (`StepFailed`) without it.

**Methods**

- `@property def holding(self) -> str | None` — The name it gave what it holds (written to that step's state as `HELD`); none while it holds nothing.
- `def close(self) -> None` — End what it keeps running (its processes, and what they hold).

### `Result`

*class* · `libraries/rollout-train/src/rollout_train/record.py`

```python
class Result
```

| Field | Type | Default | Description |
|---|---|---|---|
| `group` | `int` | required | The group's number in the run, from 1. |
| `time` | `float` | required | When it was written, in seconds since the epoch. |
| `task` | `str` | required | The row's key. |
| `title` | `str` | `''` |  |
| `rollout_seconds` | `float` | `0.0` | From the group's decision to its last episode's end (when its result was written). |
| `rewards` | `list[float]` | `field(default_factory=list[float])` | Of the episodes fit to train on, as are `solved` and `durations`. |
| `solved` | `list[bool]` | `field(default_factory=list[bool])` |  |
| `durations` | `list[float \| None]` | `field(default_factory=list[float \| None])` |  |
| `failed` | `int` | `0` | Episodes that did not complete, or asked to be left out. |
| `failures` | `list[str]` | `field(default_factory=list[str])` |  |
| `notes` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | What the algorithm said of the group. |
| `segments_recorded` | `int` | `0` |  |
| `segments` | `int` | `0` | Segments the algorithm found to train on: none if it skipped the group. |
| `skipped` | `str \| None` | `None` | Why the algorithm found nothing to train on, if it did not. |
| `unlocked` | `int` | `0` | Rows of the environment unlocked after this group. |

**Methods**

- `@classmethod def of(cls, episodes: Sequence['Episode'], *, group: int, task: str, title: str, **more: Any) -> 'Result'` — A group's result, written now, from its episodes: the rewards, `solved` and durations of those fit to train
  on, and how many did not complete and why; `more` says the rest.
- `def to_json(self) -> dict[str, Any]` — The record as the `results` table keeps it: without what the group's own record and key say (`JOINED`).
- `@classmethod def from_json(cls, data: Mapping[str, Any], number: int, group: Mapping[str, Any]) -> 'Result'` — A result as it is kept, with what its group's record (`group`, under `number`) says.

### `results`

*function* · `libraries/rollout-train/src/rollout_train/record.py`

```python
async def results(ledger: Ledger, run: str = 'train') -> list[Result]
```

How a run's groups went, by their numbers.

### `Retention`

*class* · `libraries/rollout-train/src/rollout_train/checkpoints.py`

```python
class Retention
```

Which of a run's checkpoints keep their files, their weights and their trainer state (what can be served, and
what a step can go on from): the newest `recent`, and every `every`-th by depth, so that saves thin out with
age. A blob put in the last `grace` seconds is not deleted yet (`Checkpoints.thin`): `grace` must be longer than
a checkpoint takes from its first file's put to its append, and a thinning from reading what is named to its
last delete, together.

| Field | Type | Default | Description |
|---|---|---|---|
| `recent` | `int` | `2` |  |
| `every` | `int` | `20` |  |
| `grace` | `float` | `3600.0` |  |

**Methods**

- `def kept(self, depths: list[int]) -> set[int]`

### `Schedule`

*class* · `libraries/rollout-train/src/rollout_train/evals.py`

```python
class Schedule
```

Evals a training run makes of its own checkpoints: `suite` (one version) played by the checkpoint of every
`every`th step, `episodes` episodes of each start (none: each entry's), between that step and the next. `run` gives
the eval's run for a step, and each part's: runs the run's episode runners play. `environments` are the entries'
environments, by `module:name` (any not given are imported), and `binding` how an environment's episodes are played
(by default every slot from the trained channel); each entry's limits are its own. `named` is what the run's
settings call the suite (`evals.suite`): its name (by default), which follows its newest version, or the version's
id, which does not.

| Field | Type | Default | Description |
|---|---|---|---|
| `suite` | `Suite` | required |  |
| `run` | `EvalRuns` | required |  |
| `every` | `int` | `1` |  |
| `episodes` | `int \| None` | `None` |  |
| `environments` | `Mapping[str, Environment]` | `field(default_factory=dict[str, Environment])` |  |
| `binding` | `Callable[[Environment], RunBinding] \| None` | `None` |  |
| `named` | `str \| None` | `None` |  |

**Methods**

- `def due(self, checkpoint: Checkpoint, run: str) -> bool` — Whether `checkpoint` is evaluated: a checkpoint `run` made at a step the schedule names.

### `Serving`

*class* · `libraries/rollout-train/src/rollout_train/serving.py`

```python
class Serving
```

That a run's channel serves a checkpoint from now on (or, with no checkpoint, the base model).

| Field | Type | Default | Description |
|---|---|---|---|
| `channel` | `str` | required | The channel's name within the run. |
| `checkpoint` | `str \| None` | `None` | By id; None: the model the channel's engines are started with. |
| `depth` | `int` | `0` | The checkpoint's depth: the version its samples are stamped with. |
| `kind` | `str` | `'lora'` | `lora`, an adapter; `full`, weights loaded in place of the engines' own. |
| `files` | `Manifest \| None` | `None` | What the engines load: the checkpoint's weights, or the files a bridge made of them. |
| `layout` | `str \| None` | `None` | The bridge that made `files` (`rollout_train.bridges`), by name, if one did. |
| `over` | `str \| None` | `None` | For an adapter over a full checkpoint, that checkpoint, by id: the engines hold its weights first. |
| `model` | `str \| None` | `None` | The model the channel's line began from, by name. |
| `sequence` | `int \| None` | `None` | The longest turn the trainer can train on (`Limits.sequence`), for every runner that samples the channel. |
| `max_lag` | `int \| None` | `None` | How many checkpoints behind this a sample may be, where the run says (0 for an eval, which plays one checkpoint); None: as the runner's channel says. |
| `at` | `float` | `field(default_factory=lambda: round(time.time(), 1))` |  |

**Methods**

- `def to_json(self) -> dict[str, JsonValue]`
- `@classmethod def from_json(cls, data: Mapping[str, Any]) -> 'Serving'`

### `StateLost`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class StateLost(Exception)
```

A step's full state will never be kept: its checkpoint stays incomplete, and a step from it needs the trainer
that holds it.

### `Step`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Step
```

| Field | Type | Default | Description |
|---|---|---|---|
| `metrics` | `Mapping[str, float]` | required |  |
| `keeping` | `bool` | `False` | Whether the trainer keeps the rest of the step's state itself after returning (`Keeps.kept`): `into/state` then holds only what it wrote before it returned. |

### `StepFailed`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class StepFailed(Exception)
```

A step did not produce weights: the policy is as it was, and a later step may succeed.

### `Suite`

*class* · `libraries/rollout-train/src/rollout_train/evals.py`

```python
class Suite
```

One version of a suite: its entries, each what every subject of it plays, start for start, and how.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `entries` | `list[SuiteEntry]` | required |  |
| `made` | `float` | `0.0` |  |
| `number` | `int` | `1` |  |
| `edited_from` | `int \| None` | `None` | The version it was edited from, by number. |

**Methods**

- `@property def id(self) -> str`
- `@property def environments(self) -> list[str]` — Its entries' environments, in order.
- `@property def starts(self) -> list[Start]` — Every start, numbered from 1 in this order: the first entry's, then the next's.
- `@property def held_out(self) -> bool` — Whether every entry is held out of training.
- `def entry(self, environment: str) -> SuiteEntry | None` — Its entry of an environment, if it has one.
- `def record(self) -> dict[str, Any]` — What the ledger keeps of it (its name and number are its table and key).
- `def configured(self) -> tuple[Any, ...]` — What makes an eval of it what it is: each entry's configuration, in order.

### `suite_entry`

*function* · `libraries/rollout-train/src/rollout_train/evals.py`

```python
def suite_entry(environment_name: str, environment: Environment, *, rows: Sequence[str] | None = None, seeds: Sequence[int] = (), starts: Sequence[Start] | None = None, eval_data: str | None = None, episodes: int = 1, thinking_tokens: int | None = None, answer_tokens: int | None = None) -> SuiteEntry
```

An entry of `environment` (named `environment_name`), its starts chosen as `chosen_starts` says, with `episodes`
of each start and the limits its episodes take. Raises `ValueError` for a row the environment does not have, no
starts, or a count that is no whole number of 1 at least; `KeyError` for eval data the environment does not
have.

### `suite_for`

*function* · `libraries/rollout-train/src/rollout_train/evals.py`

```python
async def suite_for(ledger: Ledger, reference: str, environment_name: str, environment: Environment) -> Suite
```

The version of a suite a reference says (`NAME`: the one its name points to; `NAME@NUMBER`: that one): the
ledger's, whatever its environments, or else `environment`'s eval data of that name, frozen now as its version 1 (on
first use). Raises `KeyError` when neither has it.

### `suite_of`

*function* · `libraries/rollout-train/src/rollout_train/evals.py`

```python
async def suite_of(ledger: Ledger, reference: str) -> Suite | None
```

The version of a suite a reference says, if there is one: `NAME@NUMBER`, that version; `NAME`, the one the
registry points the name to, else its newest.

### `SuiteEntry`

*class* · `libraries/rollout-train/src/rollout_train/evals.py`

```python
class SuiteEntry
```

One environment of a suite's version: what every subject plays of it, start for start, and how.

| Field | Type | Default | Description |
|---|---|---|---|
| `environment` | `str` | required | The environment, as `module:name`. |
| `starts` | `list[Start]` | required | Its starts, in order. |
| `environment_version` | `str \| None` | `None` | The environment's version when the entry was made. |
| `chosen` | `str` | `DRAWN` | How its starts were chosen: `EVAL_DATA`, `DRAWN` or `GIVEN`. |
| `eval_data` | `str \| None` | `None` | The name of the environment's eval data its starts are, where they are. |
| `rows` | `list[str] \| None` | `None` | The rows it names, by key. |
| `seeds` | `list[int] \| None` | `None` |  |
| `held_out` | `bool` | `False` | Whether every start is one of the environment's eval starts, which training never draws. |
| `episodes` | `int` | `1` | Episodes of each start an eval plays, unless it is asked for another number. |
| `thinking_tokens` | `int \| None` | `None` | Its episodes' tokens of thinking per turn, and of answer after it; none: the channel's own (which may be no budget). |
| `answer_tokens` | `int \| None` | `None` |  |

**Methods**

- `@property def limits(self) -> dict[str, int]` — The sampling limits it gives its episodes, by a channel's names for them (`thinking_tokens`,
  `answer_tokens`), where it gives any.
- `def configured(self) -> tuple[Any, ...]` — What makes it what it is in an eval: the environment and its version, the starts, the episodes and the
  limits.

### `train`

*function* · `libraries/rollout-train/src/rollout_train/loop.py`

```python
async def train(environment: Environment, trainer: Trainer, checkpoints: Checkpoints, *, start: str | None = None, base: str | None = None, channel: str, directory: Path, publish: Publisher, run: str = 'train', algorithm: Algorithm | None = None, groups: int = 100, groups_per_step: int = 4, groups_ahead: int | None = None, max_lag: int = MAX_LAG_DEFAULT, episodes_at_once: int = 6, seed: int = 0, binding: RunBinding | None = None, curriculum: Curriculum | None = None, retention: Retention | None = None, started: Mapping[str, JsonValue] | None = None, hooks: Sequence[Hooks] = (), kept: Callable[[], Awaitable[Collection[str]]] | None = None, made: Callable[[Checkpoint], Awaitable[object]] | None = None, reshard: Callable[[Checkpoint, Fence], Awaitable[Manifest]] | None = None, evals: Schedule | None = None, desired: Callable[[], Awaitable[Mapping[str, JsonValue]]] | None = None, scheduled: Callable[[str, int, int | None], Awaitable[Schedule | None]] | None = None) -> None
```

Train from `start` (a checkpoint's id; else the base model, named `base`) on `environment` until `groups` more
groups have been played (those a stopped loop left unplayed among them) and every group played has been trained on,
serving each checkpoint made on `channel`; a run started again goes on from the newest checkpoint it made. A step is
taken over the groups queued once at least `groups_per_step` have something to train on (and, at the end, over what
is left). `directory` is where checkpoints' files are kept on this machine while they are in use: the one being
served and the one before it (a turn in progress finishes under the weights it began with); every checkpoint's files
are in the blob store; `publish` serves a checkpoint on `channel`. `algorithm` is the one the family of the
trainer's objective takes, unless given (`rollout_train.algorithm.algorithm_for`).
`episodes_at_once` is how many episodes the run keeps work waiting for, whatever groups they are of (runners play
them, as many at once as each has places). `binding` says how the program's model slots and imports are served (by
default: every slot from `channel`, each import from the tool set of its own name). `curriculum` is one that has
recorded nothing (by default the environment's own, else the generic one: `curriculum_of`): the run's results are
folded into it. Each group's start is drawn with `train_start`, never one of the environment's eval starts.
`retention` says which of the checkpoints the run made keep their files (weights and trainer state) once a newer one
is served (`Retention()` unless given); besides those, what is served, what any run starts from, and whatever `kept`
says (the bookmarked checkpoints, say) keep theirs. `started` is what the run's `starts` record says beside what the
loop knows (where it starts from, this host, the time): where the run's directory is, where the monitor on its
machine serves (`address`), and its settings, say. `hooks` are told of each result and step; `made` is
called with each checkpoint made, once it is served (to move a bookmark, say). `reshard` gives the files the engines
load for a checkpoint (made by a bridge: `rollout_train.bridges`), told the run's fence to note it under; without
it, they load the trainer's. `evals` says which checkpoints the run evaluates as it makes them, between their step
and the next. `desired` reads what is wanted of the run's changeable settings (`rollout_train.settings`:
`groups_per_step`, `evals.…`, `trainer.…`), each time a step is about to be decided; `scheduled` makes the schedule
of evals they name (a suite by name or a version by id, every, episodes; None for a suite the run cannot play),
without which only `evals`' suite can be played. A suite named by its name is played in the version its name points
to when each step is decided: the step's record says which (`suite_version`).

### `Trained`

*class* · `libraries/rollout-train/src/rollout_train/record.py`

```python
class Trained
```

What was done with a group: the step that covered it, and the checkpoint that step made or why it failed.

| Field | Type | Default | Description |
|---|---|---|---|
| `step` | `int` | required |  |
| `checkpoint` | `str \| None` | `None` | By id, once made. |
| `error` | `str \| None` | `None` |  |

### `trained`

*function* · `libraries/rollout-train/src/rollout_train/record.py`

```python
async def trained(ledger: Ledger, run: str = 'train') -> dict[int, Trained]
```

For each group a step covers: that step, and its outcome if it has one.

### `Trainer`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Trainer(Protocol)
```

A trainer keeps nothing between steps that it cannot be given again: a step says what it starts from and
where its files go, so any trainer can take any step of any policy.

| Field | Type | Default | Description |
|---|---|---|---|
| `budget` | `Budget` | required |  |
| `weights` | `str` | required | What its steps make: `lora` (an adapter over the weights the engines hold) or `full` (all the weights). |

**Methods**

- `async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step` — Train on the batch, starting from `parent` (None: from the base model). The new weights are left in
  `into/weights`, and what a later step starts from in `into/state`. Raises `StepFailed` if the step
  produced no weights.

### `wanted`

*function* · `libraries/rollout-train/src/rollout_train/serving.py`

```python
async def wanted(ledger: Ledger, run: str, channel: str) -> Serving | None
```

What a run's channel should serve now: the record of the greatest depth it serves (`serving_of`; the newest
among equals); None if there is none (the base model).

### `Weighted`

*class* · `libraries/rollout-train/src/rollout_train/trainer.py`

```python
class Weighted
```

A segment to train on, and its advantage: every token the policy sampled in it counts by that much.

| Field | Type | Default | Description |
|---|---|---|---|
| `segment` | `Segment` | required |  |
| `advantage` | `float` | required |  |
| `source` | `str` | `''` | Where the segment is from, for the record of what a step trained on: `RUN/GROUP/EPISODE/SLOT/INDEX`. |

## `rollout_train.inference`

Channels: trainable models being served, and what they ask of an engine.

### `Channel`

*class* · `libraries/rollout-train/src/rollout_train/inference/channel.py`

```python
class Channel
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `engines` | `Sequence[Engine]` | required |  |
| `renderer` | `'Renderer'` | required | The model family's token format. |
| `limits` | `Limits` | `Limits()` |  |
| `adapter` | `str \| None` | `None` | The adapter sampling now (None: the weights the engines hold, the model's own or a full checkpoint's). |
| `serving` | `str \| None` | `None` | What is served, by name: the adapter, or the full checkpoint the engines hold (None: the model's own). |
| `version` | `int` | `0` | How many times weights have been published; recorded with every sampled token. |
| `held` | `str \| None` | `None` | The full checkpoint the engines hold, by name (None: the model's own). |
| `keep` | `int` | `MAX_LAG + 1` | Adapters kept loaded: the one served and those before it a turn may still sample from (a run's `max_lag + 1`). |
| `model` | `str \| None` | `None` | The model the engines were started with, by the name a request asks for it (`resolved`). |

**Methods**

- `@property def context_limit(self) -> int` — The longest turn the channel takes, and what it tells programs.
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, session: str = '', version: int | None = None, request: str | None = None, top: int = 0) -> Generation` — Sample from one of the engines: the same one for a session every time, where its prompts' shared
  beginnings are cached. `version` and `request` (the version the caller stamps the tokens with, and a name for
  the request) are for samplers elsewhere: this process's own callers read what it publishes.
- `async def sample(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], name: str | None, session: str = '', top: int = 0) -> Generation` — Sample what is served under `name` (a checkpoint's id, or the model's name; None: the model), as a server
  elsewhere is asked (`rollout_train.inference.remote.CheckpointServer`): `NotLoaded` where it is not served here
  once a load in progress has ended (a turn caught by a full checkpoint's load is sampled again). The answer
  names what sampled it.
- `async def score(self, tokens: Sequence[int], *, start: int, end: int | None = None, top: int = 0, adapter: str | None, session: str = '', version: int | None = None, request: str | None = None) -> Scores` — Score tokens on the session's engine (`Engine.score`), as `generate` samples there.
- `async def scored(self, tokens: Sequence[int], *, start: int, end: int | None = None, top: int = 0, name: str | None, session: str = '') -> Scores` — Score tokens with what is served under `name`, as `sample` samples it. The answer names what scored them.
- `def resolved(self, name: str | None) -> str | None` — The adapter the engines are asked for to sample what is served under `name`: the adapter itself, or None
  for the full checkpoint they hold or the model's own (`model`, or None); `NotLoaded` for anything else.
- `async def weights(self, session: str) -> tuple[str | None, int]` — The adapter a session's next turn samples from (None: the weights the engines hold), and the version its
  tokens are stamped with: the same for every session, as this process publishes to every engine at once.
- `@property def loaded(self) -> list[str]` — The adapters loaded on the engines, oldest first: the one served, and those before it (`keep` in all).
- `def adapters(self) -> list[tuple[str, int, bool]]` — What the engines hold for this channel, oldest first: each adapter, and the full checkpoint held, by name,
  with the version it was published as and whether it is full weights.
- `async def publish(self, adapter: str, path: str, version: int | None = None, *, full: bool = False) -> int` — Serve `adapter` from now on: a LoRA directory every engine can read at `path`, or with `full`, a full
  checkpoint's weights there, which the engines load in place of what they hold. Returns the version it is
  served as: `version` if one is given (the checkpoint's depth, which means the same in every process), or
  one more than the last. The adapters before stay loaded, `keep` in all with this one, so that a turn in
  progress finishes under the weights it began with; older ones are dropped. Full weights replace the engines'
  at once, and the adapters trained on the weights before go with them. Publishing what is being served
  changes nothing.
- `async def keeping(self, keep: int) -> None` — Keep `keep` adapters loaded from now on (at least one), dropping the oldest past it.
- `async def dropped(self) -> None` — Remove every adapter of this channel from the engines (other channels' on the same engines stay).
- `def forget(self, names: Collection[str]) -> None` — Take it that the engines no longer hold the adapters `names` (a server elsewhere that started again): they
  are loaded again when they are next published.
- `async def pause(self) -> None` — Hold new requests back, and wait for those in flight to finish.
- `def resume(self) -> None`
- `async def sleep(self) -> None`
- `async def wake(self) -> None`
- `def take(self) -> dict[str, float]` — What passed through since the last call: requests, tokens in and out, and throughput.
  `tokens_per_second` is everything generated over the time the channel was generating.
- `def close(self) -> None`

### `CheckpointServer`

*class* · `libraries/rollout-train/src/rollout_train/inference/remote.py`

```python
class CheckpointServer(Protocol)
```

What a `RemoteChannel` samples on: a server that holds checkpoints by name and samples the one a request names.
`RemoteEngine` is one (a vLLM server elsewhere); `rollout_train.inference.hosts.HostServer` is another (an engine
host actor).

| Field | Type | Default | Description |
|---|---|---|---|
| `address` | `str` | required | How it is known: a URL, or an actor's name. |
| `max_model_len` | `int` | required | The longest sequence it accepts, as it last said (`models`); 0 until it has. |

**Methods**

- `async def models(self, within: float = 2.0) -> dict[str, Any]` — The checkpoints it holds, by name (each a card as vLLM's `/v1/models` lists it); `Unreachable` if it does
  not answer `within` seconds.
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, session: str = '', request: str | None = None, top: int = 0) -> Generation` — Sample from the checkpoint `adapter` names (None: the model it started with); `NotLoaded` where it does not
  hold it, `Unreachable` where it does not answer. The answer names what sampled it (`Generation.model`). With
  `top`, each sampled token comes with the `top` most likely tokens there (`Engine.generate`).
- `async def score(self, tokens: Sequence[int], *, start: int, end: int | None = None, top: int = 0, adapter: str | None, session: str = '', request: str | None = None) -> Scores` — Score tokens with the checkpoint `adapter` names (`Engine.score`), refused as `generate` refuses. The answer
  names what scored them (`Scores.model`).
- `def close(self) -> None`

### `Connection`

*class* · `libraries/rollout-train/src/rollout_train/inference/remote.py`

```python
class Connection
```

How servers are reached: a bearer token read from an environment variable (`token_env`) or a file
(`token_file`), never written down; TLS verified against a CA bundle (`ca`), and a client certificate and its key
(`certificate`, `key`) for servers that ask for one. Nothing: plain HTTP, no token.

| Field | Type | Default | Description |
|---|---|---|---|
| `token_env` | `str \| None` | `None` |  |
| `token_file` | `str \| None` | `None` |  |
| `ca` | `str \| None` | `None` |  |
| `certificate` | `str \| None` | `None` |  |
| `key` | `str \| None` | `None` |  |
| `identity` | `str \| None` | `None` | The URI SAN the server's certificate must carry (`spiffe://rollout/pod/NAME`), checked in the TLS handshake in place of the host name: a server whose certificate names another identity, or none, is refused before anything is sent to it. A connection with an identity reaches only `https://` addresses: a request to any other is refused before it is sent (`httpx.UnsupportedProtocol`). None: the host name is checked, as TLS does. |

**Methods**

- `def token(self) -> str | None`
- `def client(self, timeout: float = 600.0) -> httpx.AsyncClient` — A client that reaches servers so.

### `Engine`

*class* · `libraries/rollout-train/src/rollout_train/inference/channel.py`

```python
class Engine(Protocol)
```

One replica serving a model: in this process, or a client of a server elsewhere.

| Field | Type | Default | Description |
|---|---|---|---|
| `max_model_len` | `int` | required | The longest sequence (prompt and completion) it accepts. |

**Methods**

- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, top: int = 0) -> Generation` — Sample a completion of `prompt` from `adapter` (None: the weights it holds). With `top`, each sampled
  token comes with the `top` most likely tokens there and their logprobs.
- `async def score(self, tokens: Sequence[int], *, start: int, end: int | None = None, top: int = 0, adapter: str | None) -> Scores` — The logprobs `adapter` gives the tokens at positions `start` to `end` (None: to the end) of `tokens`, each
  given those before it, with the `top` most likely tokens at each. Nothing is sampled.
- `async def load_adapter(self, name: str, path: str) -> None` — Register a LoRA adapter under `name`; requests name it to sample from it.
- `async def remove_adapter(self, name: str) -> None`
- `async def load_weights(self, path: str) -> None` — Serve the full weights in `path` (a checkpoint's files) in place of the model's own, from now on.
- `async def sleep(self) -> None` — Free the accelerator (for a trainer that shares it).
- `async def wake(self) -> None`
- `@property def processes(self) -> Sequence[int]` — The processes it started on this machine, for whoever must end them if this process is killed.
- `def close(self) -> None`

### `Generation`

*class* · `libraries/rollout-train/src/rollout_train/inference/channel.py`

```python
class Generation
```

| Field | Type | Default | Description |
|---|---|---|---|
| `tokens` | `list[int]` | required |  |
| `logprobs` | `list[float]` | required | Of each sampled token, under the distribution it was sampled from. |
| `finish_reason` | `str` | required | `stop` (a stop token, included in `tokens`) or `length`. |
| `model` | `str \| None` | `None` | The model that sampled it, where a server elsewhere says (`rollout_train.inference.remote`): the checkpoint, by the name it is served as. The gateway checks that it is the checkpoint it stamps the tokens with. |
| `top_tokens` | `list[list[int]]` | `field(default_factory=list[list[int]])` | Where a request asked for the `top` most likely tokens at each position: at each sampled token, those tokens, most likely first, under the distribution it was sampled from (empty where none were asked for). |
| `top_logprobs` | `list[list[float]]` | `field(default_factory=list[list[float]])` | Their logprobs, beside `top_tokens`. |

### `Limits`

*class* · `libraries/rollout-train/src/rollout_train/inference/channel.py`

```python
class Limits
```

What a turn may take, in tokens: the deployment's hardware decides, and code above it receives the outcome
(a context limit in a model's capability contract, a refusal when a context is full), never these numbers.

| Field | Type | Default | Description |
|---|---|---|---|
| `thinking` | `int \| None` | `None` | Tokens of thinking per turn before it is closed by force; none: thinking runs until the model closes it, or until what the context leaves after the answer's room is spent. |
| `answer` | `int \| None` | `None` | Room for the answer after the thinking; none: whatever room the turn has left. With neither budget, a turn is one generation that may fill what the context leaves (`rollout_train.recorder.sampling`). |
| `sequence` | `int \| None` | `None` | The longest turn (prompt and completion): the smaller of what the engines accept and what the trainer can train on. A long prompt leaves less room to think, so that every turn can be trained on. |

### `NotLoaded`

*class* · `libraries/rollout-train/src/rollout_train/inference/channel.py`

```python
class NotLoaded(Unserved)
```

The server does not have the model a request names (yet).

### `RemoteChannel`

*class* · `libraries/rollout-train/src/rollout_train/inference/remote.py`

```python
class RemoteChannel
```

One run's channel, sampled on servers elsewhere: what the gateway samples from (`Sampler`).

Each turn asks for the checkpoint the run says the channel should serve (`wanted`), by its id as the model's name;
or, where its server does not have it yet, the newest one before it that the server has, no more than `max_lag`
checkpoints behind (`Serving.max_lag`, where the run says, as an eval does: 0). A server that answers that it does
not have the model is asked for the one before, and for the newest again at the next look. A turn waits while no
server has a checkpoint close enough, for `patience` seconds at most (then `NoReplica`). Every token is stamped with
the depth of the checkpoint its answer names; an answer that names another model is refused, and the turn sampled
again.

The channel's servers are one URL (a router, a proxy, or a single server), or a list: a session's turns then go to
one of those that answer and have a checkpoint close enough, worked out from the session's id alone, so that nothing
is kept per session. A server is a URL (a vLLM server, reached as `connection` says) or any `CheckpointServer` (an
engine host's, say), which stays the caller's to close. What the run says and what each server has are asked again
every `every` seconds. With `discover`, the servers themselves are asked for again at each look (pods that come and
go: `rollout_train.pods.routing.LeasedServers`), beside those given.

**Methods**

- `def __init__(self, name: str, renderer: 'Renderer', limits: Limits, *, model: str, servers: Sequence['str | CheckpointServer'], wanted: Callable[[], Awaitable[Sequence[Serving]]], max_lag: int = MAX_LAG, connection: Connection | None = None, every: float | None = None, patience: float = 300.0, discover: Callable[[], Awaitable[Sequence['CheckpointServer']]] | None = None) -> None`
- `@property def limits(self) -> Limits` — The limits it was given; the longest turn the trainer can train on, as the run says, unless they say one.
- `@property def context_limit(self) -> int`
- `@property def bound(self) -> int` — How many checkpoints behind what the channel should serve a sample may be.
- `def name_of(self, said: Serving) -> str` — The model a server serves a checkpoint as: its id; the base model's name for none.
- `def choices(self) -> list[Serving]` — The checkpoints a turn may sample from now, newest first: what the channel should serve, and those before it
  no more than `bound` behind. A server that cannot serve a full checkpoint under its own name (a vLLM server)
  never lists it, so it is never offered there.
- `def offered(self, address: str) -> Serving | None` — What a server would sample a turn from now: the newest of the `choices` it has.
- `def server_of(self, session: str) -> str | None` — The server a session's turns go to now (none while none has a checkpoint close enough): of those that do,
  the one a hash of the session and its address ranks first.
- `async def refresh(self, *, now: bool = False) -> None` — Ask again what the channel should serve and what each server has (unless that was asked within `every`
  seconds and `now` is not said). A server that does not answer is given no turn.
- `async def reaches(self) -> bool` — Whether a server would take a turn now.
- `async def weights(self, session: str) -> tuple[str | None, int]` — The checkpoint a session's next turn samples from (None: the base model) and the version its tokens are
  stamped with: its depth.
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, session: str = '', version: int | None = None, request: str | None = None, top: int = 0) -> Generation` — Sample on the session's server, from the checkpoint `adapter` names. `Unserved` if the server does not have
  it any more (`NotLoaded`), answers for another, or does not answer (`Unreachable`).
- `async def score(self, tokens: Sequence[int], *, start: int, end: int | None = None, top: int = 0, adapter: str | None, session: str = '', version: int | None = None, request: str | None = None) -> Scores` — Score tokens on the session's server with the checkpoint `adapter` names, refused as `generate` is: every
  token scored is counted as a token in, and none as a token out.
- `def servers(self) -> list[dict[str, JsonValue]]` — The servers it samples on, as last asked: each one's address, the checkpoint it would sample from now and its
  depth, and how far that is behind what the channel should serve.
- `def take(self) -> dict[str, float]` — What passed through since the last call, as `Channel.take` counts it.
- `def close(self) -> None`

### `RemoteEngine`

*class* · `libraries/rollout-train/src/rollout_train/inference/remote.py`

```python
class RemoteEngine
```

An engine served elsewhere: a vLLM OpenAI-compatible server at `address` (or a router or a proxy in front of
several), serving `model` under its own name. `Engine` over its API: a request names the adapter it samples from as
its model (the base model's name for none), and an answer that names another is refused (`Unserved`), as is a model
the server does not have (`NotLoaded`). Adapters are loaded and removed by name; the server must allow it
(`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`) and read the path given on its own machine. Full weights cannot be served
under a name of their own by the server: `load_weights` refuses.

| Field | Type | Default | Description |
|---|---|---|---|
| `processes` | `Sequence[int]` | `()` |  |

**Methods**

- `def __init__(self, model: str = '', *, address: str, connection: Connection | None = None, client: httpx.AsyncClient | None = None, max_model_len: int | None = None) -> None`
- `async def models(self, within: float = 2.0) -> dict[str, Any]` — The models the server has (`/v1/models`), by name; `Unreachable` if it does not answer `within` seconds.
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, session: str = '', request: str | None = None, top: int = 0) -> Generation` — Complete the prompt's tokens with the model `adapter` names (the base model for none): the tokens sampled,
  the logprob of each, how it ended, and the model the server says sampled it. With `top`, the `top` most likely
  tokens at each, by id (`return_tokens_as_token_ids`).
- `async def score(self, tokens: Sequence[int], *, start: int, end: int | None = None, top: int = 0, adapter: str | None, session: str = '', request: str | None = None) -> Scores` — The logprobs the model `adapter` names (the base model for none) gives the tokens at positions `start` to
  `end` of `tokens`, with the `top` most likely tokens at each, and the model the server says scored them: a
  completion of the tokens up to `end` with `prompt_logprobs`, whose one generated token (vLLM generates at least
  one) is dropped. The server caps `top` at its `--max-logprobs`.
- `async def load_adapter(self, name: str, path: str) -> None` — Load the adapter at `path` (read on the server's machine) under `name`. One the server holds under that name
  already (loaded before a follower started again) is taken as loaded.
- `async def remove_adapter(self, name: str) -> None`
- `async def load_weights(self, path: str) -> None`
- `async def sleep(self) -> None` — Free the server's accelerator (it must allow it: `VLLM_SERVER_DEV_MODE=1`).
- `async def wake(self) -> None`
- `def close(self) -> None`

### `Route`

*class* · `libraries/rollout-train/src/rollout_train/inference/remote.py`

```python
class Route
```

How a channel whose engines serve elsewhere is sampled: its model family's renderer, its limits, its base
model's name, its servers (each the URL of a router, a proxy or a server, or a `CheckpointServer`, such as an engine
host's `HostServer`), how far behind a sample may be (`max_lag`), and how URLs are reached (`connection`).

| Field | Type | Default | Description |
|---|---|---|---|
| `renderer` | `'Renderer'` | required |  |
| `model` | `str` | required |  |
| `servers` | `tuple['str \| CheckpointServer', ...]` | required |  |
| `limits` | `Limits` | `field(default_factory=Limits)` |  |
| `max_lag` | `int` | `MAX_LAG` |  |
| `connection` | `Connection` | `field(default_factory=Connection)` |  |
| `discover` | `Callable[[str], Callable[[], Awaitable[Sequence['CheckpointServer']]]] \| None` | `None` | Given a run, what its channel asks at each look for servers that come and go (RunPod's pods). |

### `Routes`

*class* · `libraries/rollout-train/src/rollout_train/inference/remote.py`

```python
class Routes
```

The routed channels of every run a runner plays (`RemoteChannel`), each made when first asked for, choosing from
what that run says its channel serves (in `ledger`): what the gateway samples them through.

**Methods**

- `def __init__(self, routes: Mapping[str, Route], ledger: Ledger, *, every: float | None = None, patience: float = 300.0) -> None`
- `def routed(self, channel: str) -> bool`
- `def channel(self, run: str, channel: str) -> RemoteChannel`
- `async def reaches(self, run: str, channel: str) -> bool`
- `def channels(self) -> dict[str, RemoteChannel]` — Every routed channel made so far, by its name within its run (`RUN/NAME`).
- `def close(self) -> None`

### `Sampler`

*class* · `libraries/rollout-train/src/rollout_train/inference/channel.py`

```python
class Sampler(Protocol)
```

What the gateway samples from: a `Channel`, whose engines this process publishes to, or a channel sampled on
servers elsewhere (`rollout_train.inference.remote.RemoteChannel`).

**Methods**

- `@property def name(self) -> str`
- `@property def renderer(self) -> 'Renderer'`
- `@property def limits(self) -> Limits`
- `@property def context_limit(self) -> int`
- `async def weights(self, session: str) -> tuple[str | None, int]` — The adapter (the checkpoint) a session's next turn samples from (None: the weights the engines hold), and
  the version its tokens are stamped with.
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, session: str = '', version: int | None = None, request: str | None = None, top: int = 0) -> Generation` — Sample from the checkpoint `adapter` names, stamped `version`; `Unserved` where it is not served (or the
  server is gone). `request` names the request, for whatever logs it. With `top`, each sampled token comes with
  the `top` most likely tokens there (`Engine.generate`).
- `async def score(self, tokens: Sequence[int], *, start: int, end: int | None = None, top: int = 0, adapter: str | None, session: str = '', version: int | None = None, request: str | None = None) -> Scores` — Score the tokens at positions `start` to `end` of `tokens` with the checkpoint `adapter` names
  (`Engine.score`), as `generate` samples from it.

### `Scores` {#rollout_traininferencescores}

*class* · `libraries/rollout-train/src/rollout_train/inference/channel.py`

```python
class Scores
```

A model's logprobs of given tokens: of each position from `start` on, the logprob of the token there given the
tokens before it, and the `top` most likely tokens there with their logprobs, most likely first. Scores are of the
model's own distribution (temperature 1).

| Field | Type | Default | Description |
|---|---|---|---|
| `start` | `int` | required | The first position scored (at least 1: the first token has nothing before it). |
| `logprobs` | `list[float]` | required | Of the token at each position scored, in order. |
| `top_tokens` | `list[list[int]]` | `field(default_factory=list[list[int]])` | At each position scored, the most likely tokens, most likely first (empty where none were asked for). |
| `top_logprobs` | `list[list[float]]` | `field(default_factory=list[list[float]])` | Their logprobs, beside `top_tokens`. |
| `model` | `str \| None` | `None` | The model that scored them, where a server elsewhere says, as `Generation.model`. |

**Methods**

- `@property def end(self) -> int` — The position after the last one scored.

### `Unserved`

*class* · `libraries/rollout-train/src/rollout_train/inference/channel.py`

```python
class Unserved(Exception)
```

The checkpoint a turn began with is not served where it is asked for (not loaded yet, dropped, or another
answered), or the server cannot be reached: the turn is sampled again, from what is served then.

## `rollout_train.inference.hosts`

Engine hosts: a replica's engines as a Ray actor, serving runs by checkpoint.

### `EngineHost`

*class* · `libraries/rollout-train/src/rollout_train/inference/hosts.py`

```python
class EngineHost
```

One replica's engines and the follower that keeps them serving what the runs bound to it should: started as a
Ray actor (`started`), named `name`. `ledger_at` and `blobs_at` say where the ledger and the blob store are (as
`rollout_train.ledger.opened` and `rollout_train.stores.opened` read them); `engine` (`module:name`) makes the
engine with `model` and `options`. `bound` is the runs' channels it serves at first, `(run, channel)`. `replica`
is its index among the replicas of what it serves, and how many there are. It keeps checkpoints' files under
`directory`, and looks at what to serve every `every` seconds.

**Methods**

- `def __init__(self, name: str, ledger_at: Mapping[str, Any], blobs_at: Mapping[str, Any], engine: str, model: str, options: Mapping[str, JsonValue] | None = None, *, bound: Collection[Binding] = (), replica: tuple[int, int] = (0, 1), directory: str = SCRATCH, every: float = 2.0, beating: float = 15.0) -> None`
- `async def about(self) -> Mapping[str, JsonValue]` — What its beats say of it: its machine, its model, and when it started.
- `async def bind(self, run: str, channel: str) -> None` — Serve a run's channel too, from the next look on.
- `async def unbind(self, run: str, channel: str) -> None` — Serve a run's channel no more: its adapters are removed at the next look.
- `async def bound(self) -> list[Binding]` — The runs' channels it serves, `(run, channel)`.
- `async def follow(self) -> bool` — Look at what to serve now, as the follower does every few seconds; whether anything changed.
- `async def models(self) -> dict[str, Any]` — What it holds, by name, each as a card of vLLM's `/v1/models`: the model it was started with (unless full
  weights replaced it), each adapter (its `parent` the model), each full checkpoint, with the depth each was
  published as.
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, session: str = '', request: str | None = None, top: int = 0) -> Generation` — Sample the checkpoint `adapter` names (None: the model), as a `CheckpointServer`; `NotLoaded` where it does
  not hold it (once a load in progress has ended).
- `async def score(self, tokens: Sequence[int], *, start: int, end: int | None = None, top: int = 0, adapter: str | None, session: str = '', request: str | None = None) -> Scores` — Score tokens with the checkpoint `adapter` names (None: the model), as a `CheckpointServer`, refused as
  `generate` is.
- `async def served(self) -> list[JsonValue]` — What each channel serves, as its beats say (`Follower.served`).
- `async def pause(self) -> None` — Hold new requests back, and wait for those in flight to finish.
- `async def resume(self) -> None`
- `async def sleep(self) -> None` — Free the GPU (for a trainer that shares it); requests should be held back first (`pause`).
- `async def wake(self) -> None`
- `async def close(self) -> None` — Stop following and end the engines (before the actor is ended).

### `host_spec`

*function* · `libraries/rollout-train/src/rollout_train/inference/hosts.py`

```python
def host_spec(cluster: Cluster, provider: str, model: str, *, settings: 'RunSettings | None' = None) -> HostSpec
```

An engine host of `provider`'s `model` (an `[inference.NAME]` of the cluster, of a kind its engines run in an
engine host, `vllm`): its kind's engine (or the provider's `engine`, `module:name`), the model's options (its
context as `max_model_len`, unless they say one) with the provider's `max_logprobs` (what it declares as its top-k
logprobs, and so what its engines allow), a replica's GPUs, and `[placement.engines]`. A run whose trainer shares
the provider's card (`colocate_with`, by its `settings`) has its host ask for half of the replica's GPUs, and its
trainer for the other half.

### `HostPausable`

*class* · `libraries/rollout-train/src/rollout_train/inference/hosts.py`

```python
class HostPausable
```

`rollout_train.colocated.Pausable` over an engine host actor's handle: a colocated trainer holds the host's
requests back and puts its engines to sleep while it steps.

**Methods**

- `def __init__(self, handle: Any) -> None`
- `async def pause(self) -> None`
- `def resume(self) -> None`
- `async def sleep(self) -> None`
- `async def wake(self) -> None`

### `HostServer`

*class* · `libraries/rollout-train/src/rollout_train/inference/hosts.py`

```python
class HostServer
```

A `CheckpointServer` over an engine host actor's handle, for a `RemoteChannel` in the Ray cluster it runs in:
each call is an actor call. A host that does not answer (it died, or is starting again) is `Unreachable`.

**Methods**

- `def __init__(self, handle: Any, address: str) -> None`
- `async def models(self, within: float = 2.0) -> dict[str, Any]`
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, session: str = '', request: str | None = None, top: int = 0) -> Generation`
- `async def score(self, tokens: Sequence[int], *, start: int, end: int | None = None, top: int = 0, adapter: str | None, session: str = '', request: str | None = None) -> Scores`
- `def close(self) -> None`

### `HostSpec`

*class* · `libraries/rollout-train/src/rollout_train/inference/hosts.py`

```python
class HostSpec
```

What starts an engine host of a provider's model: its engine (`module:name`), model and options, and what it
asks Ray for.

| Field | Type | Default | Description |
|---|---|---|---|
| `engine` | `str` | required |  |
| `model` | `str` | required |  |
| `options` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` |  |
| `gpus` | `float` | `0.0` | GPUs it asks for: a fraction shares a card. |
| `resources` | `Mapping[str, float]` | `field(default_factory=dict[str, float])` | Custom resources it asks for (`[placement.engines]`), which steer it to the nodes that have them. |
| `cpus` | `float` | `1.0` | CPUs it asks for (what a run's demand counts for it: `rollout_train.demand`). |

### `started`

*function* · `libraries/rollout-train/src/rollout_train/inference/hosts.py`

```python
def started(name: str, spec: HostSpec, ledger_at: Mapping[str, Any], blobs_at: Mapping[str, Any], *, bound: Collection[Binding] = (), replica: tuple[int, int] = (0, 1), namespace: str | None = None, detached: bool = False, directory: str = SCRATCH, every: float = 2.0, beating: float = 15.0, placement: Mapping[str, Any] | None = None) -> Any
```

An engine host started as a Ray actor named `name` on the cluster this process is connected to, asking for
what `spec` says and started again whenever it dies; its handle. A run's own host goes with the job that started it,
in its bundle of the run's placement group (`placement`: the options `rollout_train.demand.placed` gives); a pool's
is `detached`, and lives until it is ended.

## `rollout_train.inference.api`

Channels on hosted APIs: by message, never trained on, spend counted.

### `ApiChannel`

*class* · `libraries/rollout-train/src/rollout_train/inference/api.py`

```python
class ApiChannel
```

One model of a hosted API, as a channel (the module's docstring). It serves no checkpoint: every turn is the
model's own, and none is trained on (`sampled_with` is empty).

| Field | Type | Default | Description |
|---|---|---|---|
| `sampled_with` | `tuple[str, ...]` | `()` |  |

**Methods**

- `def __init__(self, name: str, provider: Hosted, model: str, limits: Limits | None = None, *, endpoint: HostedEndpoint | None = None, attempts: int = ATTEMPTS, backoff: float = BACKOFF, longest_wait: float = LONGEST_WAIT) -> None`
- `@property def context_limit(self) -> int`
- `@property def max_output_tokens(self) -> int` — The most a reply may take: the model's `max_output_tokens` option, else its context.
- `@property def held(self) -> str` — What every turn is recorded as served by: the model.
- `def dollars(self, usage: Usage) -> float | None` — What a reply with this usage cost (`priced`, at the model's catalog prices).
- `async def weights(self, session: str) -> tuple[str | None, int]` — The model's own weights, at depth 0: a hosted model serves no checkpoint.
- `async def reaches(self) -> bool` — Always: whether the API answers is known once it is asked.
- `async def refresh(self) -> None` — Nothing to learn: the catalog says what the model takes.
- `def endpoint(self) -> HostedEndpoint` — The endpoint the provider names for the model, made the first time, with the key read then. Raises
  `ModelEndpointError` where it cannot be made (no key, an endpoint that does not import).
- `async def sample(self, request: SampleRequest, *, temperature: float = 1.0, top_p: float = 1.0, thinking: int | None = None, answer: int | None = None) -> SampleResult` — One reply, with the binding's sampling and budgets (none: the channel's), asked for again while the API
  cannot give it now, up to `attempts` times (the module's docstring).
- `def take(self) -> dict[str, float]` — What passed through since the last call: requests, tokens in and out, throughput, the retries and the
  dollars spent.
- `def close(self) -> None` — Nothing to end: the endpoint's connections close with the process.

### `ATTEMPTS`

*constant* · `libraries/rollout-train/src/rollout_train/inference/api.py`

```python
ATTEMPTS = 6
```

Times a reply the API cannot give now is asked for, before the channel gives up.

### `Hosted`

*class* · `libraries/rollout-train/src/rollout_train/inference/api.py`

```python
class Hosted
```

An `api` provider as a gateway reaches it: what makes its models' endpoints (`module:name`), its key, its
catalog, its base URL, and its concurrency cap with what holds it (`admission`, shared by its channels).

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `endpoint` | `str` | required |  |
| `key` | `'Secret \| None'` | required |  |
| `models` | `Mapping[str, 'ModelOffer']` | required |  |
| `base_url` | `str \| None` | `None` |  |
| `concurrency` | `int \| None` | `None` |  |
| `admission` | `asyncio.Semaphore \| None` | `field(default=None, repr=False)` |  |

**Methods**

- `@classmethod def of(cls, provider: 'InferenceProvider') -> 'Hosted'` — A cluster's `api` provider: its key is its `api_key_env` (or `api_key_file`), else its auth's key or
  token.

### `HostedEndpoint`

*class* · `libraries/rollout-train/src/rollout_train/inference/api.py`

```python
class HostedEndpoint(Protocol)
```

What samples a hosted model: an endpoint that takes the sampling parameters of each request.

**Methods**

- `async def sample(self, request: SampleRequest, *, sampling: SamplingParameters | None = None) -> SampleResult`

### `priced`

*function* · `libraries/rollout-train/src/rollout_train/inference/api.py`

```python
def priced(usage: Usage, cost: Mapping[str, float]) -> float | None
```

What a reply cost, in dollars, from its usage at a model's catalog prices (dollars per million tokens of
`input`, `cached_input`, `output` and `thinking`); none where the catalog prices nothing or the usage counts no
tokens.

## `rollout_train.recorder`

What recording a trainable channel takes: renderers, the thinking budget, segments.

### `BEHAVIOUR`

*constant* · `libraries/rollout-train/src/rollout_train/recorder/segments.py`

```python
BEHAVIOUR = ('token_exact', 'sampled_logprobs')
```

What a segment's turns must have been sampled with for it to be trained on with an importance weight.

### `ChatTemplateRenderer`

*class* · `libraries/rollout-train/src/rollout_train/recorder/renderers.py`

```python
class ChatTemplateRenderer
```

Renders with the tokenizer's chat template; parses with a family's tool-call and thinking formats.

`end` ends an assistant turn, and so do `stops` (a family whose model stops to wait for a tool's response, say).
`options` are passed to the chat template (`enable_thinking`, say). `opens` is appended to every generation
prompt: for a family whose template leaves the thinking block for the model to open, a renderer whose
`ThinkingFormat` says the prompt opens it opens it here.

**Methods**

- `def __init__(self, name: str, tokenizer: Tokenizer, tool_calls: ToolCallFormat, thinking: ThinkingFormat | None, end: str, *, stops: Sequence[str] = (), options: Mapping[str, Any] | None = None, opens: str = '') -> None`
- `def render(self, messages: Sequence[Message], tools: Sequence[ToolSpecification]) -> list[int]`
- `def encode(self, text: str) -> list[int]`
- `def decode(self, tokens: Sequence[int]) -> str`
- `def stop_token_ids(self) -> list[int]`
- `def thinking_end_token_ids(self) -> list[int]`
- `def parse(self, completion: Sequence[int], tools: Sequence[ToolSpecification]) -> Message`

### `JsonToolCalls`

*class* · `libraries/rollout-train/src/rollout_train/recorder/renderers.py`

```python
class JsonToolCalls
```

`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (Qwen3, Hermes).

| Field | Type | Default | Description |
|---|---|---|---|
| `CALL` |  | `re.compile('<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>', re.DOTALL)` |  |

**Methods**

- `def parse(self, text: str, tools: Sequence[ToolSpecification]) -> tuple[str, list[ToolCall]]`

### `rendered` {#rollout_trainrecorderrendered}

*function* · `libraries/rollout-train/src/rollout_train/recorder/renderers.py`

```python
def rendered(factory: object, model: str) -> bool | None
```

Whether a function that makes renderers renders `model`, as it says (`renders`); none where it says nothing.

### `Renderer`

*class* · `libraries/rollout-train/src/rollout_train/recorder/renderers.py`

```python
class Renderer(Protocol)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `thinking` | `ThinkingFormat \| None` | required |  |

**Methods**

- `def render(self, messages: Sequence[Message], tools: Sequence[ToolSpecification]) -> list[int]` — The prompt: every message, then the generation prompt for the assistant's next turn.
- `def encode(self, text: str) -> list[int]`
- `def decode(self, tokens: Sequence[int]) -> str` — The text of tokens, special tokens and all: what `encode` reads back.
- `def stop_token_ids(self) -> list[int]` — Tokens that end an assistant turn.
- `def thinking_end_token_ids(self) -> list[int]` — Tokens that end thinking (to stop a thinking phase on), or none if it is not a single token.
- `def parse(self, completion: Sequence[int], tools: Sequence[ToolSpecification]) -> Message` — A sampled turn as a canonical assistant message.

### `renders`

*function* · `libraries/rollout-train/src/rollout_train/recorder/renderers.py`

```python
def renders[F: Callable[..., object]](pattern: str) -> Callable[[F], F]
```

Say which models a function that makes renderers renders: those whose name matches `pattern` anywhere, case
ignored (`r"qwen3\.5"`: `Qwen/Qwen3.5-9B`, `cyankiwi/Qwen3.5-9B-AWQ-4bit`).

### `sample_turn`

*function* · `libraries/rollout-train/src/rollout_train/recorder/sampling.py`

```python
async def sample_turn(request: SampleRequest, renderer: 'Renderer', limits: Limits, context_limit: int, generate: Generate) -> SampledTurn
```

Sample one reply to `request`. Raises `ContextOverflow` when the prompt leaves no room to answer.

### `Segment`

*class* · `libraries/rollout-train/src/rollout_train/recorder/segments.py`

```python
class Segment
```

A piece of a session's trajectory: tokens that only grew, as the policy saw and continued them.

| Field | Type | Default | Description |
|---|---|---|---|
| `tokens` | `list[int]` | required |  |
| `spans` | `list[Span]` | required |  |
| `logprobs` | `list[float]` | required | Behavior logprobs of the tokens inside the spans, in order. |
| `channel` | `str` | `''` | The channel that sampled them. The spans' `version`s are the depths of the checkpoints it served. |
| `sampled_with` | `tuple[str, ...]` | `TOKEN_LEVEL` | What every one of its turns was sampled with, of `TOKEN_LEVEL`. |
| `trained` | `bool` | `True` | Whether it may be trained on: false for a segment of a slot that is not trained (a judge's, a fixed opponent's), which is kept for what it shows and what it cost, and never trained on. |
| `teacher` | `TeacherScores \| None` | `None` | A teacher's scores of its sampled tokens, where a teacher scored them (for distillation); none otherwise. |

**Methods**

- `@property def sampled(self) -> int`
- `@property def lacks(self) -> tuple[str, ...]` — What its turns were sampled without, of `BEHAVIOUR` (empty: it can be trained on with an importance
  weight).

### `segments_of`

*function* · `libraries/rollout-train/src/rollout_train/recorder/segments.py`

```python
def segments_of(turns: Sequence[Turn], *, untrained: Collection[str] = ()) -> list[Segment]
```

The segments of a session's turns, oldest first, each with at least one sampled span. What the turns named in
`untrained` (by effect id) sampled is kept as context, with no span: it is not trained on.

### `Span`

*class* · `libraries/rollout-train/src/rollout_train/recorder/segments.py`

```python
class Span
```

Tokens `start` to `end` (exclusive) of a segment were sampled by the policy, at weights `version` (the depth
of the checkpoint served then).

| Field | Type | Default | Description |
|---|---|---|---|
| `start` | `int` | required |  |
| `end` | `int` | required |  |
| `version` | `int` | required |  |
| `effect_id` | `str` | `''` | The sample that produced them: the `effect_id` its run's events know it by. |

### `TeacherScores`

*class* · `libraries/rollout-train/src/rollout_train/recorder/segments.py`

```python
class TeacherScores
```

A teacher's scores of a segment's sampled tokens, one for each, in the order of its spans (as
`Segment.logprobs`): its logprob of the token, and the teacher's most likely tokens there with their logprobs,
most likely first (`rollout_train.distillation.teacher_scores` makes them from a teacher's `Scores`).

| Field | Type | Default | Description |
|---|---|---|---|
| `teacher` | `str` | required | The channel that scored them. |
| `logprobs` | `list[float \| None]` | required | Of each sampled token, given the tokens before it: none for one the teacher did not score (beyond its context). |
| `top_tokens` | `list[list[int]]` | `field(default_factory=list[list[int]])` | At each sampled token, the teacher's most likely tokens (at most the top-k asked for; fewer where it gave fewer, none where it did not score the token), or empty where no top-k was asked for. |
| `top_logprobs` | `list[list[float]]` | `field(default_factory=list[list[float]])` | Their logprobs, beside `top_tokens`. |

**Methods**

- `@property def top(self) -> int` — The most tokens any position carries (0: no top-k).

### `ThinkingFormat`

*class* · `libraries/rollout-train/src/rollout_train/recorder/renderers.py`

```python
class ThinkingFormat
```

How a family delimits thinking. Its generation prompt may already open the block.

| Field | Type | Default | Description |
|---|---|---|---|
| `open` | `str` | required |  |
| `close` | `str` | required |  |
| `prompt_opens` | `bool` | required | Whether the generation prompt ends inside an opened thinking block (the model only closes it). |
| `forced_close` | `str` | required | Text appended to end thinking that ran out of budget (masked from training). |

### `TOKEN_LEVEL`

*constant* · `libraries/rollout-train/src/rollout_train/recorder/segments.py`

```python
TOKEN_LEVEL = ('token_exact', 'sampled_logprobs', 'honours_sampling')
```

What a turn can be sampled with: its tokens are the exact ones sampled, each sampled token's behaviour logprob is
known, and temperature and top-p were applied (`rollout_train.providers.Capabilities`). vLLM and Tinker sample with
all three, and a turn that does not say what it was sampled with was sampled by one of them.

### `tokenizer_of`

*function* · `libraries/rollout-train/src/rollout_train/recorder/renderers.py`

```python
def tokenizer_of(model: str) -> Tokenizer
```

The tokenizer of a checkpoint, by its name or path.

### `ToolCallFormat`

*class* · `libraries/rollout-train/src/rollout_train/recorder/renderers.py`

```python
class ToolCallFormat(Protocol)
```

How a family writes tool calls in its output.

**Methods**

- `def parse(self, text: str, tools: Sequence[ToolSpecification]) -> tuple[str, list[ToolCall]]` — Split text into what precedes the calls and the calls; arguments are converted to their schema types.

### `XmlFunctionCalls`

*class* · `libraries/rollout-train/src/rollout_train/recorder/renderers.py`

```python
class XmlFunctionCalls
```

`<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` (Qwen3.5, Qwen3-Coder).

| Field | Type | Default | Description |
|---|---|---|---|
| `CALL` |  | `re.compile('<tool_call>\\s*<function=([^>\\s]+)>(.*?)</function>\\s*</tool_call>', re.DOTALL)` |  |
| `PARAMETER` |  | `re.compile('<parameter=([^>\\s]+)>\\n?(.*?)\\n?</parameter>', re.DOTALL)` |  |

**Methods**

- `def parse(self, text: str, tools: Sequence[ToolSpecification]) -> tuple[str, list[ToolCall]]`

## `rollout_train.gateway`

The stateless gateway: samples channels for harnesses and records every turn.

### `Attempt`

*class* · `libraries/rollout-train/src/rollout_train/gateway/client.py`

```python
class Attempt
```

What a program's run plays, as the keys of its slots say.

| Field | Type | Default | Description |
|---|---|---|---|
| `run` | `str` | required | The run whose tables its turns go under. |
| `fence` | `Fence` | required | What its turns are appended under. |
| `episode` | `str` | `''` | `GROUP/EPISODE`. |
| `attempt` | `int` | `0` |  |

### `ChannelDirectory`

*class* · `libraries/rollout-train/src/rollout_train/gateway/directory.py`

```python
class ChannelDirectory
```

Every run's channels, built from its start when the run is first asked for (`load`), over the servers of the
providers in `providers`, rendered with the renderer each channel names (`renderers`, given its `module:name` and
the model; by default the renderer itself, called with the model), and over the hosted APIs in `hosted`. What each
channel should serve and what its servers have are asked again every `every` seconds; a turn waits up to
`patience` seconds for a server with a checkpoint close enough.

**Methods**

- `def __init__(self, ledger: Ledger, providers: Mapping[str, Provided], *, hosted: Mapping[str, Hosted] | None = None, pods: Mapping[str, 'Auth'] | None = None, tls: 'Tls | None' = None, renderers: Callable[[str, str], 'Renderer'] = _renderer, every: float | None = None, patience: float = 300.0) -> None`
- `@classmethod def of(cls, cluster: 'Cluster', ledger: Ledger, **options: Any) -> 'ChannelDirectory'` — A directory over the cluster config's providers whose servers answer vLLM's API at their endpoints
  (`SERVED_AT_ENDPOINTS`), each reached as its auth says, and its `api` providers.
- `async def load(self, run: str) -> dict[str, Built]` — A run's channels, by name, built from its newest start the first time (a run whose start names no
  provider this directory knows has none; one with no start yet is asked again next time).
- `def channel(self, run: str, name: str) -> Built | None` — A run's channel, once the run is loaded.
- `def channels(self) -> dict[str, Built]` — Every channel built so far, by its name within its run (`RUN/NAME`).
- `def close(self) -> None`

### `create_app`

*function* · `libraries/rollout-train/src/rollout_train/gateway/service.py`

```python
def create_app(gateway: Gateway) -> Starlette
```

Serve `gateway` over HTTP (behind a proxy that terminates TLS, or with uvicorn's own certificates).

### `Gateway`

*class* · `libraries/rollout-train/src/rollout_train/gateway/service.py`

```python
class Gateway
```

What a replica serves: where it records, the keys it takes, and the channels it samples: every channel a run's
start names with a provider `directory` knows (`rollout_train.gateway.directory`), those whose engines this process
publishes to (`channels`, by name; `models` names each one's base model), and those whose engines serve elsewhere
(`routes`), each run's sampled from what that run says it serves. `hooks` are told of each sample a
harness asks for in one of the three APIs and the gateway records (a runner's own samples reach its hooks through
its endpoints). Each turn records what its sampler samples with: the sampler's `sampled_with` where it says, else
`TOKEN_LEVEL` (every engine and server a channel samples from is token-exact, with sampled-token logprobs). The
channels on hosted APIs it samples by name are `hosted`; what runs spend on them is counted in `spending`.

| Field | Type | Default | Description |
|---|---|---|---|
| `store` | `TurnStore` | required |  |
| `keyring` | `Keyring` | required |  |
| `channels` | `Mapping[str, Channel]` | `field(default_factory=dict[str, Channel])` |  |
| `routes` | `Routes \| None` | `None` |  |
| `models` | `Mapping[str, str]` | `field(default_factory=dict[str, str])` |  |
| `hooks` | `Sequence[RunHooks]` | `()` |  |
| `directory` | `ChannelDirectory \| None` | `None` |  |
| `hosted` | `Mapping[str, ApiChannel]` | `field(default_factory=dict[str, ApiChannel])` |  |
| `spending` | `Spending \| None` | `None` |  |

**Methods**

- `async def granted(self, key: str) -> Grant` — The grant a key carries, its run's channels loaded from its start (where there is a directory); `Refused`
  if it is not one this gateway takes.
- `async def load(self, run: str, channel: str) -> None` — Build a run's channels from its start, if there is a directory and they are not built yet (a channel named
  within its run, `RUN/NAME`, is that run's).
- `def sampler(self, grant: Grant) -> 'Sampler | ApiChannel'` — What a grant's turns sample from (`sampler_of` its run and channel).
- `def sampler_of(self, run: str, channel: str) -> 'Sampler | ApiChannel'` — What a run's channel samples from: the channel its start names, built by the directory (once the run is
  loaded: `load`); else the channel of this process it names (its engines here, or a hosted API); else the run's
  routed channel of that name. A channel named within its run (`RUN/NAME`) is that run's.
- `async def reaches(self, run: str, channel: str) -> bool` — Whether a run's channel can be sampled now: one its start names or a routed one, whose servers have a
  checkpoint close enough to what the run says it should serve; or one of this process.
- `@property def names(self) -> list[str]` — The channels it samples, by name.
- `def describe(self, grant: Grant) -> CapabilityContract`
- `async def sample(self, grant: Grant, request: SampleRequest, links: Sequence[Link] = ()) -> Reply` — One reply, recorded before it is returned: the one recorded under the request's effect id, if there is one.
  `links` are those its harness declared besides the request's own. Raises `Refused`, or the endpoint's
  `ModelEndpointError` (`ContextOverflow` when the context is too long).
- `async def score(self, grant: Grant, request: ScoreRequest) -> Reply` — The scores the grant's channel gives the request's tokens (`Reply.scores`), recorded as a turn of its own
  use before they are returned: those recorded under the request's effect id, if there are any. Raises `Refused`
  (`ValueError` from the channel, for a range or a `top` it does not take, is a request refused), or the
  endpoint's `ModelEndpointError` (`ContextOverflow` for a sequence too long to score).
- `def observe(self, grant: Grant, request: SampleRequest, reply: Reply, seconds: float) -> None` — Tell the hooks of a sample a harness asked for, newly recorded.
- `async def count(self, grant: Grant, prompt: Prompt) -> int` — How many tokens a prompt renders to with the channel's renderer: what a turn's prompt would hold. A hosted
  API's channel has no renderer, and counts none (`Refused`).
- `async def ready(self) -> dict[str, str]` — What is not ready, by part (empty: ready): the ledger and the blob store must answer.

### `GatewayEndpoint`

*class* · `libraries/rollout-train/src/rollout_train/gateway/client.py`

```python
class GatewayEndpoint
```

Implements `AddressableEndpoint` for one recorded binding, through the gateway.

**Methods**

- `def __init__(self, endpoints: GatewayEndpoints, binding: RecordedModel) -> None`
- `def describe(self, session_id: str) -> CapabilityContract`
- `def address(self, session_id: str) -> ModelAddress` — The gateway, and a key for the session. What a harness samples there is recorded by the gateway (a gateway
  in this process tells the runner's hooks of it).
- `async def cancel(self, effect_id: str) -> None` — Nothing to do: a turn whose sampling is cancelled in this process is not recorded, and a gateway elsewhere
  records the turn whether or not it is awaited.
- `async def sample(self, request: SampleRequest) -> SampleResult`

### `GatewayEndpoints`

*class* · `libraries/rollout-train/src/rollout_train/gateway/client.py`

```python
class GatewayEndpoints
```

Implements `RecordedEndpoints` over a gateway: the one in this process (`gateway`), else the one at `url` (its
base URL, without `/v1`), with keys signed by `keyring`; `store` is the turn store the gateway records in. What a
channel guarantees is the gateway's to say, in this process; else, for a routed channel, what `routes` say of it
(the runner's view of the same servers), and for any other, `contracts` (for a channel the gateway at `url` hosts,
what it says of it: `hosted`). `url` is also what a harness is handed:
without one, a run cannot give a harness an address.

**Methods**

- `def __init__(self, url: str | None, keyring: Keyring, store: TurnStore, contracts: Mapping[str, CapabilityContract] | None = None, *, gateway: Gateway | None = None, routes: Routes | None = None, lifetime: float = LIFETIME, http: httpx.AsyncClient | None = None, retries: int = 5, backoff: float = 0.5) -> None`
- `@classmethod def of(cls, gateway: Gateway, url: str | None = None, *, lifetime: float = LIFETIME) -> 'GatewayEndpoints'` — Endpoints over a gateway in this process; `url`, where it is also served over HTTP, for harnesses.
- `@property def http(self) -> httpx.AsyncClient`
- `@property def channels(self) -> list[str]` — The channels it samples, by name.
- `async def hosted(self, channels: Collection[str], *, patience: float = 60.0) -> None` — Learn what the gateway at the URL guarantees of each of `channels`, which its own engines serve (its
  `/v1/models` says each one's contract), asking again while it cannot be reached, for up to `patience` seconds.
  Raises `ModelEndpointError` when it cannot be reached, or does not host one of them.
- `def admit(self, run_id: str, attempt: Attempt) -> None` — Say which attempt a program's run plays, before it starts (and again when the attempt is taken up anew).
- `def forget(self, run_id: str) -> None`
- `def endpoint(self, binding: RecordedModel) -> 'GatewayEndpoint'`
- `def attempt(self, run_id: str) -> Attempt` — The attempt an admitted run plays.
- `def key(self, session_id: str, binding: RecordedModel) -> str` — A key for a session of an admitted run.
- `def contract(self, session_id: str, binding: RecordedModel) -> CapabilityContract` — What a binding's channel guarantees a session, with the thinking and answer room the binding gives in place
  of the channel's: a routed channel, as its run's servers say (the run is admitted).
- `async def reaches(self, run: str, binding: RunBinding) -> bool` — Whether every recorded model of a run's binding can be sampled now: a channel the gateway in this process
  samples (one the run's start names, or a routed one, only once its servers have a checkpoint close enough to
  what the run says it should serve), or one the gateway elsewhere serves (a routed one likewise, as this
  process sees its servers; one the run's start names, as that gateway lists the run's channels).
- `async def sessions(self, run: str, run_id: str) -> dict[str, list[Segment]]` — What each slot of a program's run recorded, by slot.

### `Grant`

*class* · `libraries/rollout-train/src/rollout_train/gateway/keys.py`

```python
class Grant
```

What a key lets its holder do: sample for one model slot of one attempt, until it expires.

| Field | Type | Default | Description |
|---|---|---|---|
| `run` | `str` | required | The run whose tables the turns go under (a training run, an eval's run). |
| `run_id` | `str` | required | The program's run that plays the attempt: the session is `{run_id}/{slot}`. |
| `slot` | `str` | required |  |
| `channel` | `str` | required |  |
| `fence` | `Fence` | required | What its turns are appended under: a turn whose fence was taken again since is refused. |
| `expires` | `float` | required | Seconds since the epoch. |
| `episode` | `str` | `''` | `GROUP/EPISODE`, for an attempt of an episode a run asked for. |
| `attempt` | `int` | `0` |  |
| `temperature` | `float` | `1.0` |  |
| `top_p` | `float` | `1.0` |  |
| `thinking` | `int \| None` | `None` | Tokens of thinking per turn, and of answer after it, in place of the channel's own (the binding's: an eval's, say); none: the channel's. |
| `answer` | `int \| None` | `None` |  |
| `trained` | `bool` | `True` | Whether its turns may be trained on: false for a slot that is not trained (a judge, a fixed opponent). |

**Methods**

- `@property def session_id(self) -> str`
- `def to_json(self) -> dict[str, Any]`
- `@classmethod def from_json(cls, data: Mapping[str, Any]) -> 'Grant'`

### `KeyRefused`

*class* · `libraries/rollout-train/src/rollout_train/gateway/keys.py`

```python
class KeyRefused(Exception)
```

A key that is malformed, signed by no secret of the keyring, forged, or expired.

### `Keyring`

*class* · `libraries/rollout-train/src/rollout_train/gateway/keys.py`

```python
class Keyring
```

Secrets by id: `signing` signs, every one verifies.

| Field | Type | Default | Description |
|---|---|---|---|
| `secrets` | `Mapping[str, bytes]` | required |  |
| `signing` | `str` | required |  |
| `leeway` | `float` | `30.0` | Seconds a key is still taken after it expires, for clocks that differ. |

**Methods**

- `@classmethod def parse(cls, pairs: list[tuple[str, str]]) -> 'Keyring'` — Secrets as (id, secret) pairs, the signing one first.
- `@classmethod def from_environment(cls, environment: Mapping[str, str] | None = None) -> 'Keyring'` — The keyring `ROLLOUT_GATEWAY_KEYS` or `ROLLOUT_GATEWAY_KEYS_FILE` says.
- `@classmethod def load(cls, path: Path) -> 'Keyring'` — The keyring in a file: one `KID SECRET` per line, the signing one first (`#` begins a comment).
- `def mint(self, grant: Grant) -> str` — A key for `grant`, signed with the signing secret.
- `def verify(self, key: str, now: float | None = None) -> Grant` — The grant a key carries. Raises `KeyRefused` unless a secret of the keyring signed it and it has not
  expired.

### `Link`

*class* · `libraries/rollout-train/src/rollout_train/gateway/turns.py`

```python
class Link
```

What a harness said of a request: that it follows from an earlier one (`source`, by its request id: its
effect id), and how (`type`: `compaction_attempt`, `compaction`, `subagent_call`, `subagent_return`, or any other
label, which is kept as it is).

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `str` | required |  |
| `source` | `str` | required |  |

### `Provided`

*class* · `libraries/rollout-train/src/rollout_train/gateway/directory.py`

```python
class Provided
```

How a provider's servers are reached: each a URL (a vLLM server, a router in front of several) or any
`CheckpointServer`, and the connection URLs are reached with.

| Field | Type | Default | Description |
|---|---|---|---|
| `servers` | `tuple['str \| CheckpointServer', ...]` | required |  |
| `connection` | `Connection` | `field(default_factory=Connection)` |  |

### `Refused` {#rollout_traingatewayrefused}

*class* · `libraries/rollout-train/src/rollout_train/gateway/service.py`

```python
class Refused(Exception)
```

A request the gateway will not sample: why, as a `Failure`.

**Methods**

- `def __init__(self, failure: Failure, message: str) -> None`

### `Reply`

*class* · `libraries/rollout-train/src/rollout_train/gateway/turns.py`

```python
class Reply
```

What a recorded turn answered, and who it answered.

| Field | Type | Default | Description |
|---|---|---|---|
| `effect_id` | `str` | required |  |
| `slot` | `str` | required |  |
| `checkpoint` | `str` | required |  |
| `depth` | `int` | required |  |
| `result` | `SampleResult` | required |  |
| `replayed` | `bool` | `True` | Whether it was recorded before (False: by the call that returned it). |
| `timings` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | The turn's timings (`TurnRecord.timings`), as recorded. |
| `scores` | `Scores \| None` | `None` | A scoring turn's scores (`TurnRecord.scores`). |

### `ScoreRequest`

*class* · `libraries/rollout-train/src/rollout_train/gateway/service.py`

```python
class ScoreRequest(BaseModel)
```

A request to score tokens: the logprobs the channel gives the tokens at positions `start` to `end` of `tokens`
(`end` absent: to the end), each given those before it, with the `top` most likely tokens at each.

| Field | Type | Default | Description |
|---|---|---|---|
| `effect_id` | `str` | `Field(min_length=1)` | Its request id: a request under one that was recorded is answered with what was recorded. |
| `session_id` | `str` | required | The key's session. |
| `tokens` | `list[int]` | `Field(min_length=2)` |  |
| `start` | `int` | `Field(ge=1)` |  |
| `end` | `int \| None` | `None` |  |
| `top` | `int` | `Field(default=0, ge=0)` |  |

### `TurnRecord`

*class* · `libraries/rollout-train/src/rollout_train/gateway/turns.py`

```python
class TurnRecord
```

One turn, as the gateway sampled and recorded it.

| Field | Type | Default | Description |
|---|---|---|---|
| `effect_id` | `str` | required | Its request id. |
| `run` | `str` | required |  |
| `run_id` | `str` | required |  |
| `slot` | `str` | required |  |
| `channel` | `str` | required |  |
| `checkpoint` | `str` | required | What served it, by the name its endpoint answered with (a checkpoint's id, or the base model's name). |
| `depth` | `int` | required | The checkpoint's depth: the version every sampled token is stamped with. |
| `prompt` | `'array[int]'` | required |  |
| `completion` | `list[int]` | required |  |
| `mask` | `list[bool]` | required | True where the policy sampled the token; False where it was forced. |
| `logprobs` | `list[float]` | required | Behaviour logprobs of every completion token (forced ones: NaN). |
| `result` | `SampleResult` | required | The reply: the parsed message, how it finished, usage. |
| `episode` | `str` | `''` | `GROUP/EPISODE`, for an attempt of an episode a run asked for. |
| `attempt` | `int` | `0` |  |
| `links` | `tuple[Link, ...]` | `()` |  |
| `timings` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | `started` (seconds since the epoch), `phases` (seconds each generation took) and `seconds` (the whole turn). |
| `sampled_with` | `tuple[str, ...]` | `TOKEN_LEVEL` | What it was sampled with, of `TOKEN_LEVEL`: what its sampler could do (a turn recorded without saying was sampled with all of them). |
| `use` | `str` | `SAMPLE` | `sample`, or `score`: the logprobs the channel gave the prompt's tokens (`scores`), with nothing sampled. |
| `scores` | `Scores \| None` | `None` | A scoring turn's scores. |
| `trained` | `bool` | `True` | Whether it may be trained on: false for a turn of a slot that is not trained (a judge, a fixed opponent), and for a turn a hosted API sampled. |
| `spend` | `float \| None` | `None` | Dollars it cost, from its usage at its model's catalog prices, where its provider is metered and prices it. |

**Methods**

- `@property def version(self) -> int`
- `@property def session_id(self) -> str`

### `turns_table`

*function* · `libraries/rollout-train/src/rollout_train/gateway/turns.py`

```python
def turns_table(run: str, run_id: str) -> str
```

The table of a program's run's turns.

### `TurnStore`

*class* · `libraries/rollout-train/src/rollout_train/gateway/turns.py`

```python
class TurnStore
```

Turns in a ledger and a blob store. It keeps nothing that correctness depends on: the blobs it read are a
cache.

**Methods**

- `def __init__(self, ledger: Ledger, blobs: Blobs, *, chunk_tokens: int = CHUNK_TOKENS, cached: int = 4096) -> None`
- `async def index(self, run: str, run_id: str) -> dict[str, JsonValue]` — The ledger's records of a program's run's turns, by effect id, in the order they were recorded.
- `async def reply(self, run: str, run_id: str, effect_id: str, index: Mapping[str, JsonValue] | None = None) -> Reply | None` — What the turn recorded under an effect id answered, if there is one (`index`: the run's records, if read).
- `async def record(self, turn: TurnRecord, fence: Fence, index: Mapping[str, JsonValue] | None = None) -> Reply` — Keep a turn, unless one was recorded under its effect id first; returns what the turn recorded answers
  (`replayed` if it was not this one). `index` is the run's records, if they were read. Raises `Fenced` if
  `fence` was taken again.
- `async def turns(self, run: str, run_id: str) -> list[TurnRecord]` — A program's run's turns, in the order they were recorded.
- `async def sessions(self, run: str, run_id: str, *, accepted_only: bool = False) -> dict[str, list[Segment]]` — What each model slot of a program's run exports, by slot (`segments_of` its samples: scoring turns are left
  out, and so are a hosted API's, which hold no tokens). The segments of a slot that is not trained are kept,
  marked so (`Segment.trained`). With `accepted_only`, what a compaction attempt sampled is trained on only if
  its harness went on from it.

### `unaccepted`

*function* · `libraries/rollout-train/src/rollout_train/gateway/turns.py`

```python
def unaccepted(turns: Sequence[TurnRecord]) -> set[str]
```

The compaction attempts no turn went on from (by effect id): a request linked from an earlier one as its
`compaction_attempt`, and to no later one as the source of a `compaction`.

## `rollout_train.jobs`

A run's job: built from its settings and the cluster config, claiming what it needs.

### `driven`

*function* · `libraries/rollout-train/src/rollout_train/jobs.py`

```python
async def driven(launch: str, cluster: Cluster, stores: Stores | None = None) -> int
```

Run a launch's run, noting on the launch that it runs, what it waits for, and how it ended: ended, failed (with
why: its settings' refusals among them) or stopped (cancelled); a run that fails raises. A launch that finished
already is not run again: a job submitted again after its run failed (its RayJob's `backoffLimit`) returns 1, so
that the job fails too, and otherwise 0.

### `HoursReached`

*class* · `libraries/rollout-train/src/rollout_train/jobs.py`

```python
class HoursReached(LimitReached)
```

A run ran as long as its `limits.hours` allows.

### `imitated`

*function* · `libraries/rollout-train/src/rollout_train/jobs.py`

```python
def imitated(settings: RunSettings, kind: str) -> 'Objective'
```

The objective an imitate run trains with: the likelihood or preference preset its settings name (as the
dataset's `kind` holds examples, or pairs and labelled examples), else `sft` with those of their components a
likelihood takes. Raises `ValueError` for a policy-gradient preset named, or one that does not fit the dataset.

### `main`

*function* · `libraries/rollout-train/src/rollout_train/jobs.py`

```python
def main(arguments: Sequence[str] | None = None) -> None
```

`python -m rollout_train.jobs LAUNCH`: a run's job.

### `NotEnoughMemory`

*class* · `libraries/rollout-train/src/rollout_train/jobs.py`

```python
class NotEnoughMemory(Exception)
```

Stopping is better than exhausting the machine (a host may shut down rather than kill one process).

### `ran`

*function* · `libraries/rollout-train/src/rollout_train/jobs.py`

```python
async def ran(run: Run) -> None
```

Check the run's settings again, then run it as its kind says.

### `Run` {#rollout_trainjobsrun}

*class* · `libraries/rollout-train/src/rollout_train/jobs.py`

```python
class Run
```

A run being built from its settings, and what it started: everything the loop of its kind is given.

| Field | Type | Default | Description |
|---|---|---|---|
| `cluster` | `Cluster` | required |  |
| `stores` | `Stores` | required |  |
| `settings` | `RunSettings` | required |  |
| `run` | `Entry` | required |  |
| `preset` | `str \| None` | `None` | The preset its settings came from (`NAME@N`), recorded as provenance. |
| `resumes` | `bool` | `False` | Whether it goes on from where a start of it before stopped (it trains with the objective that one did). |
| `noted` | `Callable[[str], Awaitable[None]] \| None` | `None` | Told what the run waits for, as it changes (its launch's detail). |
| `environment` | `Environment \| None` | `None` |  |
| `started` | `dict[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | What the run's start records beside what its loop knows. |
| `directory` | `Path` | `field(default_factory=Path)` |  |
| `origin` | `str \| None` | `None` | The checkpoint it starts from, by id (none: the base model). |
| `trainer` | `Trainer \| None` | `None` |  |
| `hosts` | `dict[str, list[Any]]` | `field(default_factory=dict[str, list[Any]])` | The engine hosts of each channel, by channel. |
| `channels` | `dict[str, Channel]` | `field(default_factory=dict[str, Channel])` | The channels whose engines are in this process (Tinker's). |
| `on_apis` | `dict[str, ApiChannel]` | `field(default_factory=dict[str, ApiChannel])` | The channels on hosted APIs. |
| `routes` | `Routes \| None` | `None` |  |
| `gateway` | `Gateway \| None` | `None` |  |
| `recorder` | `GatewayEndpoints \| None` | `None` |  |
| `runner` | `EpisodeRunner \| None` | `None` |  |
| `feed` | `Any` | `None` |  |
| `tool_bindings` | `dict[str, ToolBinding]` | `field(default_factory=dict[str, ToolBinding])` |  |
| `pool_bindings` | `dict[str, PoolBinding]` | `field(default_factory=dict[str, PoolBinding])` |  |
| `runs` | `set[str]` | `field(default_factory=set[str])` | The runs its runner plays: its own, and its evals'. |
| `chain` | `tuple[Bridge, ...]` | `()` | The bridges the trained channel's files are made by (none: the trainer's files as they are). |
| `objective` | `Mapping[str, JsonValue] \| None` | `None` | The objective its trainer is made with, where the run's settings do not say it all (an imitate run's). |
| `waiting` | `Waiting` | `field(default_factory=Waiting)` |  |
| `trainer_handle` | `Any` | `None` |  |
| `sandboxes` | `frozenset[str]` | `frozenset()` | The kinds of sandbox its environment's programs declare. |
| `demand` | `Demand \| None` | `None` | What its scheduled parts need (`rollout_train.demand`). |
| `group` | `Any` | `None` | The placement group that reserves them. |
| `asked_at` | `float \| None` | `None` | When it asked Ray for its placement group. |
| `reserved_at` | `float \| None` | `None` | When Ray had reserved all of it (or, for a demand with no bundle, when the driver had its own). |
| `pods` | `Any` | `None` | Its pods on RunPod (`rollout_train.pods.leasing.Pods`), where its providers give it any. |
| `began` | `float` | `field(default_factory=time.time)` | When this start of it began: what `limits.hours` counts from. |

**Methods**

- `@property def ledger(self) -> Any`
- `@property def checkpoints(self) -> Checkpoints`
- `@property def kind(self) -> str`
- `@property def channel(self) -> str` — The channel the run trains or plays: the trained one, else the first its settings name.
- `async def start(self, stack: contextlib.AsyncExitStack, *, training: bool) -> None` — Start what the run needs, registering with `stack` how each is stopped: its placement group, reserved
  before anything is started in it; the trainer (with `training`), each channel's engines, the gateway, the
  pools and the runner; then wait for what Ray has yet to give.
- `def hosted(self, channel: str, provider: str, model: str) -> list[Any]` — The servers of a channel on a `vllm` provider: engine hosts of the run's own, one per replica, each bound to
  the run's channel, asking Ray for its share of a GPU and its CPU in its bundle of the run's placement group.
- `def binding(self, environment: Environment) -> Any` — How an environment's episodes are played: each slot from the channel the settings bind it to (the run's
  channel unless said), each import and pool where the cluster serves it.
- `async def publish(self, channel: str, adapter: str, files: Fetched, version: int | None = None, *, full: bool = False) -> int` — Serve a checkpoint on a channel: on engines in this process (Tinker's), its files read here and loaded now;
  elsewhere, its engine hosts and servers follow the run's serving record and read the files themselves, and this
  returns the version given.
- `async def bridged(self, checkpoint: Checkpoint, fence: Fence) -> Manifest` — The files the trained channel's engines load for a checkpoint, made by its bridges as Ray tasks.
- `def chosen(self, formats: frozenset[str]) -> tuple[Bridge, ...]` — The bridges that make what the run's channel's first provider loads from checkpoints in `formats` (none
  where the files are served as they are); `ValueError` where none does.
- `async def eval_run(self, step: int | None = None, part: int | None = None) -> str` — The run of an eval, by id, which the runner plays: with `step`, the eval of the checkpoint this run made at
  that step (`RUN-eval-STEP`, called `NAME-eval-STEP`); else this run (an eval itself). With `part`, the run that
  plays that entry of an eval of several (its id and `-PART`).
- `async def bookmarked(self) -> set[str]` — The checkpoints bookmarks name (which keep their files).
- `async def made(self, checkpoint: Checkpoint) -> None` — Carry the run's bookmark, if it names one, to a checkpoint it made.

### `run_directory`

*function* · `libraries/rollout-train/src/rollout_train/jobs.py`

```python
def run_directory(cluster: Cluster, run: str) -> Path
```

Where a run keeps its files on its driver's node: its feed, the checkpoints in use, fetched bases.

### `SpendReached`

*class* · `libraries/rollout-train/src/rollout_train/jobs.py`

```python
class SpendReached(LimitReached)
```

A run spent what its `limits.spend` allows.

### `taken_by`

*function* · `libraries/rollout-train/src/rollout_train/jobs.py`

```python
def taken_by(making: Any, settings: Mapping[str, Any]) -> dict[str, Any]
```

Those of `settings` that what makes a trainer takes: every one where it takes any keyword, else those it names
(a run's recorded settings hold every setting of its trainer's kind, which a trainer of its own may not all
take).

### `TrainerActor`

*class* · `libraries/rollout-train/src/rollout_train/jobs.py`

```python
class TrainerActor
```

A trainer in a Ray actor: made with `implementation` (`module:name`), the model and its settings, and asked
for steps by a `TrainerClient`.

**Methods**

- `def __init__(self, implementation: str, model: str, settings: Mapping[str, Any]) -> None`
- `def described(self) -> dict[str, Any]` — What the client says of the trainer: its budget, its weights, its objective, its changeable settings.
- `async def step(self, batch: list[Item], seed: int, parent: Files | None, into: Path) -> Step`
- `def change(self, settings: Mapping[str, JsonValue]) -> dict[str, JsonValue]`

### `TrainerClient`

*class* · `libraries/rollout-train/src/rollout_train/jobs.py`

```python
class TrainerClient
```

A `Trainer` over a `TrainerActor`'s handle: each step is an actor call (an error the trainer raised is raised
as itself), and what it takes between steps is changed there.

**Methods**

- `def __init__(self, handle: Any, described: Mapping[str, Any]) -> None`
- `@property def changeable(self) -> Mapping[str, JsonValue]`
- `def change(self, settings: Mapping[str, JsonValue]) -> None`
- `async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step`

## `rollout_train.demand`

What a run's scheduled parts need, and the placement group that reserves them together.

### `BRIDGE`

*constant* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
BRIDGE = 'bridge'
```

The bridge's part, by name.

### `bridge_asks`

*function* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
def bridge_asks(bridge: Bridge, cluster: Cluster) -> Resources
```

What a bridge's task asks Ray for: its declared CPUs and memory, or what `[bridges."NAME"]` says.

### `Bundle`

*class* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
class Bundle
```

One bundle of a run's placement group: the parts placed in it, and whether it is on the driver's node.

| Field | Type | Default | Description |
|---|---|---|---|
| `parts` | `tuple[Part, ...]` | required |  |
| `on_driver` | `bool` | `False` |  |

**Methods**

- `@property def resources(self) -> Resources`

### `colocating`

*function* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
def colocating(settings: 'RunSettings', cluster: Cluster) -> bool
```

Whether the run's trainer shares the GPU of its trained channel's engine hosts (`colocate_with`).

### `Demand`

*class* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
class Demand
```

What a run's scheduled parts need: its driver's (the job's entrypoint), and its placement group's bundles.

| Field | Type | Default | Description |
|---|---|---|---|
| `driver` | `Resources` | required |  |
| `bundles` | `tuple[Bundle, ...]` | `()` |  |
| `strategy` | `str` | `PACK` |  |

**Methods**

- `@property def reserved(self) -> Resources` — What its placement group reserves.
- `@property def total(self) -> Resources` — The driver's and the placement group's.
- `@property def parts(self) -> dict[str, Part]`
- `def index(self, name: str) -> int | None` — The bundle a part is placed in, by its name; none for a part the demand does not count.
- `def asks(self, name: str) -> Resources | None` — What a part asks Ray for, by its name.
- `def to_json(self) -> dict[str, JsonValue]`

### `demand`

*function* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
def demand(settings: 'RunSettings', cluster: Cluster) -> Demand
```

What a run with these settings needs on this cluster (the module's docstring). Parts the cluster does not offer
are left out (validation refuses them).

### `HEADROOM`

*constant* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
HEADROOM = Resources(cpus=0.25, memory_gib=2.0)
```

Room a pod of a run's Ray cluster keeps beyond what Ray schedules: Ray's own processes (its GCS, raylet, dashboard
and object store: measured 0.08 of a CPU at the 95th percentile, and 0.5 GiB), and some of the memory its GPU
processes hold outside Ray's count. They hold more than this (an engine host of Qwen3-0.6B 4 to 4.9 GiB, its trainer
2.2 GiB while it steps): the pod's memory limit, not its request, bounds them.

### `Part`

*class* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
class Part
```

One actor or task of a run, by name (`trainer`, `engine/CHANNEL/N`, `bridge`), and what it asks Ray for.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `asks` | `Resources` | required |  |

### `placed`

*function* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
def placed(group: Any, asked: Demand | None, part: str) -> dict[str, Any]
```

The options that place a part in its bundle of `group` (none where there is no group, or the demand does not
count the part).

### `played_channel`

*function* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
def played_channel(settings: 'RunSettings') -> str
```

The channel a run trains or plays: the trained one, else the first its settings name.

### `Pod` {#rollout_traindemandpod}

*class* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
class Pod
```

One kind of pod of a run's Ray cluster on Kubernetes: the head (`head`) or a worker group (`engines-N`), what its
parts ask for (`asked`), and how many there are.

| Field | Type | Default | Description |
|---|---|---|---|
| `group` | `str` | required |  |
| `asked` | `Resources` | required |  |
| `replicas` | `int` | `1` |  |

**Methods**

- `@property def ray(self) -> Resources` — What Ray schedules on each: whole CPUs and GPUs, as Ray starts a node with.
- `@property def requests(self) -> Resources` — What each pod asks Kubernetes for: what its parts ask for (its GPUs whole), and `HEADROOM`.

### `pods`

*function* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
def pods(asked: Demand, most: Resources | None = None, *, known: Collection[str] = ()) -> tuple[Pod, ...]
```

The pods of a run's Ray cluster: one head pod for all of it where its requests fit `most` (a pod's upper bound,
in the resources `known`: the template's limits, one node's worth); else the head holds the driver, the trainer's
bundle and the bundles with no GPU (the bridge's), and each other bundle (an engine host's) is a worker pod,
grouped by size (`engines-0`, `engines-1`, ...).

### `requested`

*function* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
def requested(asked: Demand, *, kubernetes: bool = False, most: Resources | None = None, known: Collection[str] = ()) -> Resources
```

What a run's Ray cluster asks for in all: each pod's requests (`pods`), and on Kubernetes the pod that submits
its job.

### `reserve`

*function* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
def reserve(asked: Demand, name: str) -> Any
```

A placement group for a run's demand on the Ray cluster this process is connected to, named `name`: its bundles
in order, the trainer's pinned to this process's node. Ray reserves every bundle at once or none (`ready()` says
when); none where the demand has no bundle.

### `Resources`

*class* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
class Resources
```

CPUs, memory, GPUs and custom resources (`[placement.ROLE]`), as Ray counts them.

| Field | Type | Default | Description |
|---|---|---|---|
| `cpus` | `float` | `0.0` |  |
| `memory_gib` | `float` | `0.0` |  |
| `gpus` | `float` | `0.0` |  |
| `custom` | `Mapping[str, float]` | `field(default_factory=dict[str, float])` |  |

**Methods**

- `def most(self, other: 'Resources') -> 'Resources'` — Each resource the larger of the two.
- `def beyond(self, room: 'Resources', *, known: Collection[str] = ('cpus', 'memory_gib', 'gpus')) -> list[str]` — What of this exceeds `room` (only the resources in `known`, and the custom ones `room` names), each as
  `CPUs (9 for 8)`.
- `def bundle(self) -> dict[str, float]` — As a Ray placement group bundle (memory in bytes), leaving out what is zero. A share of GPUs above one is
  rounded up to whole GPUs: Ray takes fractions of one GPU only.
- `def said(self) -> str` — In words: `1 GPU, 2 CPUs, 1 GiB`.
- `def to_json(self) -> dict[str, JsonValue]`

### `SUBMITTER`

*constant* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
SUBMITTER = Resources(cpus=0.1, memory_gib=0.25)
```

The pod KubeRay starts to submit a RayJob's job (the requests of the chart's `submitterPodTemplate`; measured: 0.06
of a CPU over a job, 130 MiB), which Kueue counts with the job's.

### `TRAINER`

*constant* · `libraries/rollout-train/src/rollout_train/demand.py`

```python
TRAINER = 'trainer'
```

The trainer's part, by name (an engine host's is `engine/CHANNEL/N`).

## `rollout_train.launching`

Asking for a run: its settings in layers, the facts validation reads, the offers.

### `capacity_of`

*function* · `libraries/rollout-train/src/rollout_train/launching.py`

```python
def capacity_of(beats: Sequence[Beat]) -> dict[str, JsonValue] | None
```

The GPUs the machines that beat now have, and those of them idle (under a twentieth of their memory used), by
machine; none where no beat says.

### `checked`

*function* · `libraries/rollout-train/src/rollout_train/launching.py`

```python
async def checked(settings: RunSettings, cluster: Cluster, ledger: Ledger, *, loaded: 'Environment | None' = None, own: str | None = None, gpus: float | None = None, free: Resources | None = None) -> list[Finding]
```

Everything wrong with a run's settings on this cluster (`rollout_train.validation.check`), with the facts
gathered now.

### `checkpoints_at`

*function* · `libraries/rollout-train/src/rollout_train/launching.py`

```python
def checkpoints_at(settings: RunSettings, cluster: Cluster, *, written: Mapping[str, Any] | None = None) -> dict[str, JsonValue] | None
```

Where a training run's checkpoints go (none for a run that makes none): `store`, the blob store its trainer
writes them to (`rollout_train.stores.described`: the store its RunPod providers name, else `[blobs]`; or the one
`written` says, where a run's start recorded it); `tinker`, whether Tinker keeps the weights (the store holds
pointers to Tinker's archive); `bridges`, the bridges that write converted copies for the trained channel's first
provider (none where its files are served as they are), and `bridged`, the store those copies go to (`[blobs]`).

### `declared`

*function* · `libraries/rollout-train/src/rollout_train/launching.py`

```python
def declared(environment: 'Environment') -> tuple[frozenset[str], EnvironmentFacts]
```

The sandbox kinds an environment's first program declares, and what validation reads of it.

### `environment_facts`

*function* · `libraries/rollout-train/src/rollout_train/launching.py`

```python
async def environment_facts(environment: str | None, cluster: Cluster, ledger: Ledger, *, loaded: 'Environment | None' = None) -> EnvironmentFacts | None
```

What validation reads of an environment: imported here (`loaded`, where the caller has it), its sandboxes, slots
and what an episode samples (its description's `turns`, `samples_per_turn`, `prompt_tokens`); a published one's
sandboxes and what an episode samples as its version recorded them, or that there is no such version. None where it
is not known here: an environment whose Python is a project of its own, which this process does not import.

### `Examined`

*class* · `libraries/rollout-train/src/rollout_train/launching.py`

```python
class Examined
```

A run's settings, checked: the findings, what is known of its environment, its estimated spend (one step's, or
an eval's), what it trains and where its checkpoints go (`checkpoints_at`).

| Field | Type | Default | Description |
|---|---|---|---|
| `findings` | `list[Finding]` | required |  |
| `environment` | `EnvironmentFacts \| None` | required |  |
| `spend` | `Spend` | required |  |
| `weights` | `str \| None` | required |  |
| `checkpoints` | `dict[str, JsonValue] \| None` | `None` |  |

### `examined`

*function* · `libraries/rollout-train/src/rollout_train/launching.py`

```python
async def examined(settings: RunSettings, cluster: Cluster, ledger: Ledger, *, loaded: 'Environment | None' = None, own: str | None = None, gpus: float | None = None, free: Resources | None = None) -> Examined
```

A run's settings checked on this cluster with the facts gathered now (`checked`), with those facts' environment,
its estimated spend (`rollout_train.validation.spend_of`), what it trains (`weights_of`) and where its checkpoints
go (`checkpoints_at`).

### `free_name`

*function* · `libraries/rollout-train/src/rollout_train/launching.py`

```python
def free_name(wanted: str, taken: set[str]) -> str
```

A name no run has: the one wanted, else it with the first number after it that no run has.

### `ledger_facts`

*function* · `libraries/rollout-train/src/rollout_train/launching.py`

```python
async def ledger_facts(settings: RunSettings, ledger: Ledger, *, own: str | None = None, gpus: float | None = None, free: Resources | None = None, cluster: Cluster | None = None) -> LedgerFacts
```

What validation reads of the ledger: each checkpoint the settings name (the start, a fixed channel's), the
suites their evals name, the names other runs have (`own`, the run's id, is left out), the GPUs and free resources
the caller knows of, and, for a run with pods on `cluster`'s RunPod providers, how long a step took here lately.

### `offers`

*function* · `libraries/rollout-train/src/rollout_train/launching.py`

```python
async def offers(cluster: Cluster, ledger: Ledger, beats: Sequence[Beat] = ()) -> dict[str, Any]
```

What a run can be asked for here (the module's docstring), as JSON.

### `ray_free`

*function* · `libraries/rollout-train/src/rollout_train/launching.py`

```python
def ray_free() -> Resources | None
```

What the Ray cluster this process is connected to has free now: CPUs, memory, GPUs and custom resources (none:
not connected). Its total is not said: an autoscaled cluster has more than its nodes now.

### `Refused` {#rollout_trainlaunchingrefused}

*class* · `libraries/rollout-train/src/rollout_train/launching.py`

```python
class Refused(ValueError)
```

A run whose settings are refused: the findings that refuse it (each with the setting it is about), and the notes
beside them.

**Methods**

- `def __init__(self, findings: Sequence[Finding]) -> None`
- `@property def refusals(self) -> list[Finding]`

### `settled`

*function* · `libraries/rollout-train/src/rollout_train/launching.py`

```python
async def settled(kind: str, name: str | None, settings: Mapping[str, JsonValue], *, preset: str | None = None, presets: Presets | None = None) -> tuple[RunSettings, str | None]
```

A run's settings: the preset's (`NAME` or `NAME@N`: those a run of its kind takes, so that a training run's
preset serves an eval of the same channels), then `settings`, then its kind and name; and the preset's version
(`NAME@N`). Raises `KeyError` for a preset there is none of.

## `rollout_train.submitting`

Starting a run's job as a Ray job or a RayJob, and reading how it goes.

### `ask`

*function* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
async def ask(settings: RunSettings, ledger: Ledger, *, preset: str | None = None, resumes: str | None = None) -> Launch
```

Record a launch of a run with these settings (its kind and name among them): of the run it resumes, else a run
registered now under its name. Raises `ValueError` (`rollout_train.registry.Taken`) for a name another run has.

### `Backend`

*class* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
class Backend(Protocol)
```

Where runs' jobs go: Ray's job API, or RayJobs on Kubernetes.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |

**Methods**

- `async def start(self, launch: Launch, entrypoint: str, runtime_env: Mapping[str, JsonValue], asked: Demand | None = None) -> str` — Start a launch's job, sized for what its run needs (`asked`); its name (a Ray job's submission id, a
  RayJob's name).
- `async def status(self, job: str) -> JobState` — How a job goes now.
- `async def stop(self, job: str) -> None` — Ask a job to stop: its driver is interrupted, and notes its run stopped.

### `backend_of`

*function* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
def backend_of(cluster: Cluster) -> Backend
```

Where a cluster's runs' jobs go: RayJobs where its config has `[kubernetes]`, else Ray's job API at `[ray]
jobs`.

### `demand_of`

*function* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
def demand_of(launch: Launch, cluster: Cluster) -> Demand
```

What a launch's run needs (`rollout_train.demand`).

### `entrypoint_of`

*function* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
def entrypoint_of(launch: Launch, cluster: Cluster) -> str
```

What a launch's job runs: `python -m rollout_train.jobs LAUNCH` in the interpreter its run starts in (its
environment's own, for one in a project's Python; the cluster's `[ray] python` otherwise).

### `followed`

*function* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
async def followed(launch: Launch, launches: Launches, cluster: Cluster | None = None, *, backends: Mapping[str, Backend] | None = None) -> Launch
```

A launch that is going, with what its job's status says noted: a job that waits (with why), runs, ended, failed
or stopped without its driver saying so. A launch whose job cannot be read is as it was.

### `job_name`

*function* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
def job_name(launch: Launch) -> str
```

The name of a launch's job: `run-` and its id, lowercase (a Kubernetes name: letters, digits and dashes).

### `JobState`

*class* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
class JobState
```

How a job goes, in a launch's states (`submitted`, `running`, `ended`, `failed`, `stopped`), with why.

| Field | Type | Default | Description |
|---|---|---|---|
| `state` | `str` | required |  |
| `detail` | `str \| None` | `None` |  |

### `KubernetesApi`

*class* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
class KubernetesApi
```

What makes, reads and deletes RayJobs, and reads Kueue's objects: the API server at `base`, with the service
account's token and CA (by default the pod's own), over `transport` where given (a test's).

| Field | Type | Default | Description |
|---|---|---|---|
| `ACCOUNT` |  | `Path('/var/run/secrets/kubernetes.io/serviceaccount')` |  |

**Methods**

- `def __init__(self, base: str = 'https://kubernetes.default.svc', *, token: str | None = None, ca: str | None = None, transport: httpx.AsyncBaseTransport | None = None) -> None`
- `async def create(self, namespace: str, resource: Mapping[str, Any]) -> dict[str, Any]`
- `async def get(self, namespace: str, name: str) -> dict[str, Any] | None`
- `async def workloads(self, namespace: str, uid: str) -> list[dict[str, Any]]` — Kueue's Workloads of a job, by the job's uid (none where Kueue is not installed).
- `async def put_secret(self, namespace: str, name: str, data: Mapping[str, bytes]) -> None` — Make the Secret `name` hold `data` (its keys and their bytes), in place of what it held, or made anew.
- `async def read(self, path: str) -> dict[str, Any] | None` — What the API server answers at `path` (`/apis/GROUP/VERSION/...`): none where it is not found. Raises
  `RuntimeError` for any other refusal (a resource the account may not read, an API that is not served).
- `async def delete(self, namespace: str, name: str) -> None`

### `POD_SECURITY`

*constant* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
POD_SECURITY = {'enforce': ('baseline', 'restricted'), 'warn': ('restricted',), 'audit': ('restricted',)}
```

The Pod Security Admission levels the release's namespace is labelled with (`pod-security.kubernetes.io/MODE`): it
enforces `baseline` (every pod of the platform passes it), and warns of and audits what `restricted` would refuse
(docs/deploy/kubernetes.md#pod-security).

### `pod_security`

*function* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
async def pod_security(section: KubernetesSection, api: KubernetesApi | None = None) -> list[str]
```

What is wrong with the Pod Security labels of `section`'s namespace, in words (`POD_SECURITY`): each label that
is missing or names another level. Raises `RuntimeError` where the namespace cannot be read.

### `RayJobResources`

*class* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
class RayJobResources
```

Runs' jobs as RayJobs in a Kubernetes namespace, each made from the cluster config's template
(`[kubernetes]`).

| Field | Type | Default | Description |
|---|---|---|---|
| `name` |  | `'kubernetes'` |  |

**Methods**

- `def __init__(self, section: KubernetesSection, api: KubernetesApi | None = None) -> None`
- `def template(self) -> dict[str, Any]`
- `async def start(self, launch: Launch, entrypoint: str, runtime_env: Mapping[str, JsonValue], asked: Demand | None = None) -> str`
- `async def status(self, job: str) -> JobState`
- `async def stop(self, job: str) -> None`

### `RayJobs`

*class* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
class RayJobs
```

Runs' jobs as Ray jobs, submitted to the job server at `address` (`client`: a `JobSubmissionClient`, or one
like it).

| Field | Type | Default | Description |
|---|---|---|---|
| `name` |  | `'ray'` |  |

**Methods**

- `def __init__(self, address: str, client: Any = None) -> None`
- `def client(self) -> Any`
- `async def start(self, launch: Launch, entrypoint: str, runtime_env: Mapping[str, JsonValue], asked: Demand | None = None) -> str`
- `async def status(self, job: str) -> JobState`
- `async def stop(self, job: str) -> None`

### `rendered` {#rollout_trainsubmittingrendered}

*function* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
def rendered(template: Mapping[str, Any], launch: Launch, entrypoint: str, runtime_env: Mapping[str, JsonValue], namespace: str, *, asked: Demand | None = None, queue: str | None = None) -> dict[str, Any]
```

The RayJob of a launch's job, made from `template` (a RayJob as YAML reads it: its Ray cluster, image, volumes,
retries): its name and labels, its entrypoint and the driver's CPUs, its runtime environment (as YAML, as KubeRay
takes it), its job's submission id and metadata. With `asked`, its Ray cluster is sized from the run's demand
(`sized`). With `queue`, it is labelled with Kueue's queue and made suspended: Kueue starts it once it admits it.
Everything else is the template's.

### `runtime_env_of` {#rollout_trainsubmittingruntime_env_of}

*function* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
async def runtime_env_of(launch: Launch, cluster: Cluster, ledger: Ledger) -> dict[str, JsonValue]
```

The Ray runtime environment of a launch's job: the cluster config as JSON (`ROLLOUT_CLUSTER_JSON`), in the
runtime environment of the published version it plays, if it plays one. Raises `KeyError` for a published version
the ledger does not keep.

### `sized`

*function* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
def sized(cluster: Mapping[str, Any], asked: Demand) -> dict[str, Any]
```

A RayJob's Ray cluster (`rayClusterSpec`) sized from a run's demand (`rollout_train.demand.pods`): its head pod
asks for the driver's and the placement group's resources and `HEADROOM`, where that fits the head's limits (one
node's worth); else the head holds the driver, the trainer's bundle and the bridge's, and each engine host's bundle
is a worker pod of a worker group made from the head's template. Ray starts each node with what its pod holds
(`rayStartParams`).

### `start`

*function* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
async def start(launch: Launch, cluster: Cluster, ledger: Ledger, backend: Backend | None = None) -> Launch
```

Start a recorded launch's job; the launch, submitted (or failed, saying why the job could not be made).

### `stopped`

*function* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
async def stopped(launch: Launch, launches: Launches, cluster: Cluster | None = None, *, backends: Mapping[str, Backend] | None = None) -> Launch
```

Ask a launch to stop: one whose job was not made yet is stopped at once; a job going is asked to stop, and its
driver notes its run stopped. Raises `KeyError` for a launch that is not going.

### `submit`

*function* · `libraries/rollout-train/src/rollout_train/submitting.py`

```python
async def submit(settings: RunSettings, cluster: Cluster, ledger: Ledger, *, preset: str | None = None, resumes: str | None = None, backend: Backend | None = None) -> Launch
```

Record a launch of a run with these settings and start its job (`ask`, then `start`), with what follows from its
settings said: what it trains and each channel's renderer (`rollout_train.validation.completed`).

## `rollout_train.launches`

Runs asked for, the jobs they became, and how each goes.

### `as_launch`

*function* · `libraries/rollout-train/src/rollout_train/launches.py`

```python
def as_launch(data: Mapping[str, Any]) -> Launch
```

A launch as it was stored. A launch asked for a profile (its `asked` names one) is read as its run settings:
its environment, start, bookmark, groups, groups a step and seed among them, and an eval's suite and episodes.

### `Asked`

*class* · `libraries/rollout-train/src/rollout_train/launches.py`

```python
class Asked
```

What a run is asked to be: its kind, its name, its run settings (as given: the schema's defaults are not
written), the preset they came from (`NAME@N`), and, for a launch that resumes a run, that run's id.

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `str` | `TRAIN` |  |
| `name` | `str` | `''` |  |
| `settings` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` |  |
| `preset` | `str \| None` | `None` |  |
| `resumes` | `str \| None` | `None` |  |

**Methods**

- `@property def environment(self) -> str | None` — The environment it plays (`module:name`, or a published one as `NAME@VERSION`), if it plays one.

### `changed`

*function* · `libraries/rollout-train/src/rollout_train/launches.py`

```python
def changed(launch: Launch, expect: Collection[str] | None, changes: Mapping[str, Any]) -> Launch | None
```

A launch with `changes`, if they may be made of it as it is (`Launches.note`); else None.

### `FileLaunches`

*class* · `libraries/rollout-train/src/rollout_train/launches.py`

```python
class FileLaunches
```

`Launches` in `launches.json` in a ledger's directory, under the lock the ledger's files are written under.

**Methods**

- `def __init__(self, directory: Path) -> None`
- `async def ask(self, asked: Asked, run: str | None = None) -> Launch`
- `async def all(self) -> list[Launch]`
- `async def note(self, id: str, *, expect: Collection[str] | None = None, **changes: Any) -> Launch`

### `Launch`

*class* · `libraries/rollout-train/src/rollout_train/launches.py`

```python
class Launch
```

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required |  |
| `asked` | `Asked` | required |  |
| `at` | `float` | required | When it was asked for. |
| `state` | `str` | `ASKED` |  |
| `run` | `str \| None` | `None` | The run it is, by id: registered when it was asked for, or the run it resumes. |
| `job` | `str \| None` | `None` | Its job: a Ray job's submission id, or a RayJob's name. |
| `backend` | `str \| None` | `None` | Where its job is: `ray` (Ray's job API) or `kubernetes` (a RayJob). |
| `detail` | `str \| None` | `None` | Why it failed, how it ended, or what its run waits for. |
| `updated` | `float` | `0.0` |  |

### `launch_of`

*function* · `libraries/rollout-train/src/rollout_train/launches.py`

```python
async def launch_of(launches: Launches, id: str) -> Launch
```

A launch by id. Raises `KeyError` when there is none.

### `Launches`

*class* · `libraries/rollout-train/src/rollout_train/launches.py`

```python
class Launches(Protocol)
```

**Methods**

- `async def ask(self, asked: Asked, run: str | None = None) -> Launch` — Record a launch, as asked, of the run `run` (by id); the launch.
- `async def all(self) -> list[Launch]` — Every launch, newest first.
- `async def note(self, id: str, *, expect: Collection[str] | None = None, **changes: Any) -> Launch` — Note how a launch goes (its state, job, detail), in one step that compares and sets: the changes are written
  only if the launch is in a state of `expect` (any, if None) and may go to the state they name (`MOVES`).
  Returns the launch as it is then, changed or not: whoever moves it compares the state it gets with the state
  it asked for. Raises `KeyError` when there is no such launch.

### `launches_of`

*function* · `libraries/rollout-train/src/rollout_train/launches.py`

```python
def launches_of(ledger: Ledger) -> Launches | None
```

The launches beside a ledger: a file beside a ledger of files, a table in a database ledger's database.

### `MOVES`

*constant* · `libraries/rollout-train/src/rollout_train/launches.py`

```python
MOVES: Mapping[str, frozenset[str]] = {ASKED: frozenset({SUBMITTED, RUNNING, FAILED, STOPPED}), SUBMITTED: frozenset({RUNNING, STOPPING, STOPPED, ENDED, FAILED}), RUNNING: frozenset({STOPPING, STOPPED, ENDED, FAILED}), STOPPING: frozenset({STOPPED, ENDED, FAILED})}
```

Where a launch may go from where it is. A launch that finished (ended, failed, stopped) goes nowhere, nothing goes
back, and a launch asked to stop is not running again: a stop asked for while its job starts stays, and its job is
stopped. A driver may start before its submitter has noted its job (asked to running). A state may also be noted again
(its details changed).

### `new_launch`

*function* · `libraries/rollout-train/src/rollout_train/launches.py`

```python
def new_launch(asked: Asked, run: str | None = None) -> Launch
```

### `OPEN`

*constant* · `libraries/rollout-train/src/rollout_train/launches.py`

```python
OPEN = (ASKED, SUBMITTED, RUNNING, STOPPING)
```

### `stored` {#rollout_trainlaunchesstored}

*function* · `libraries/rollout-train/src/rollout_train/launches.py`

```python
def stored(launch: Launch) -> str
```

A launch as JSON, as it is stored.

## `rollout_train.monitor`

A live web page over every run of a ledger.

### `FeedReader`

*class* · `libraries/rollout-train/src/rollout_train/monitor/feed.py`

```python
class FeedReader
```

Reads a feed directory incrementally: each call picks up what was appended since the last. Of a run it
keeps a summary; the run's lines are read from its file when they are asked for. Its methods may be called from
several threads at once.

**Methods**

- `def __init__(self, directory: Path) -> None`
- `def refresh(self) -> None`
- `def runs(self) -> list[dict[str, Any]]` — Every run in the feed, newest first: its labels, state, rewards and how much it has done.
- `def lines(self, run_id: str, after: int = 0) -> list[dict[str, Any]]` — A run's lines from index `after` on.

### `plain`

*function* · `libraries/rollout-train/src/rollout_train/monitor/feed.py`

```python
def plain(message: Message) -> dict[str, JsonValue]
```

A message as the page shows it: its text, its reasoning, the tools it called and the results it carries.

### `RunFeed`

*class* · `libraries/rollout-train/src/rollout_train/monitor/feed.py`

```python
class RunFeed(RunHooks, Hooks)
```

Writes every run's events and samples under `directory`, one file per run, as they happen; and what a
runner and the loop note, at the level they think at, in one file more.

`keep` bounds the directory: when more runs than that have files, the oldest are deleted. A directory has one
writer at a time: runs that an earlier writer left without an end (its process was stopped) are marked cancelled
when the next one starts, so that a monitor does not show them running for ever.

**Methods**

- `def __init__(self, directory: Path, *, keep: int = 200) -> None`
- `def on_event(self, event: RunEvent) -> None`
- `def on_note(self, event: Mapping[str, JsonValue]) -> None`
- `def on_sample(self, sample: ModelSample) -> None`
- `def close(self) -> None`

### `System`

*class* · `libraries/rollout-train/src/rollout_train/monitor/system.py`

```python
class System
```

**Methods**

- `def __init__(self, directory: Path | None = None, feed: FeedReader | None = None, *, ledger: Ledger | None = None, client: httpx.Client | None = None, importer: Importer | None = None, cluster: 'Cluster | None' = None, backends: 'Mapping[str, Backend] | None' = None) -> None` — Over a run's `directory` (its ledger, as `rollout_train.ledger.of_run` finds it: every run that shares
  it), or over a `ledger` alone. `feed` reads the directory's feed (by default its `feed`). `client` asks the
  monitors on other machines for their runs' episodes. `importer` is where environments imported from git go
  (`import_environment`); none: this monitor imports none.
- `@property def ledger(self) -> str` — Where the ledger is: its directory, or its database's URL.
- `async def snapshot(self, relayed: bool = False) -> dict[str, Any]` — Where everything stands now: every run (where it is and whether it is running; its groups that are not
  done with and the ones that are), the checkpoints (each with where it came from and the bookmarks that name it),
  the runners and what they play, what each channel serves and how fast, the machine, and what is kept.
- `async def pause(self, run: str) -> Desired` — Pause a run (`rollout_train.resuming.pause`). Raises `KeyError` where there is no such run.
- `async def resume(self, run: str, preset: str | None = None) -> Resumed` — Resume a run: in place, or by a launch of its recorded settings (over `preset`'s, for a run whose start
  records no providers: `rollout_train.resuming.resume`). Raises `Taken` for a run that cannot be resumed,
  `KeyError` where there is no such run or this monitor has no cluster config to submit on,
  `rollout_train.launching.Refused` for settings the cluster refuses.
- `async def rename(self, who: str, name: str) -> Entry` — Call the run that `who` is (its id or its name) `name` from now on, in the registry beside the ledger.
  Raises `Taken` for a name it cannot have, `KeyError` when there is no such run (or no registry).
- `async def bookmark(self, name: str, checkpoint: str) -> Bookmark` — Make a bookmark name the checkpoint `checkpoint` says (its id, the start of one, `RUN:STEP`, `RUN` or another
  bookmark), or move it there. Raises `Taken` for a name that cannot be one, `KeyError` for a reference that
  says no checkpoint (or no registry).
- `async def unbookmark(self, name: str) -> None` — Take a bookmark away (the checkpoint stays). Raises `KeyError` when there is no such bookmark.
- `async def launches(self) -> dict[str, Any]` — The runs asked for, newest first, each with the job it became and its state: a launch that is going is
  read with its job's status too (`rollout_train.submitting.followed`), so a job that waits says why, and one
  that died without its driver saying so is failed.
- `async def offers(self) -> dict[str, Any]` — What a run can be asked for here (`rollout_train.launching.offers`); nothing where this monitor was started
  without a cluster config.
- `async def check(self, body: Mapping[str, Any]) -> dict[str, Any]` — What a launch's body would be refused for, and the notes beside (each with the setting it is about), on
  this monitor's cluster (`rollout_train.launching.examined`); the settings it would run with, what it trains,
  its estimated spend on its metered parts (one step's, or an eval's: `per`; or why it cannot be estimated
  yet), where its checkpoints go (`rollout_train.launching.checkpoints_at`), and the slots its
  environment's programs declare, where they are known here. Raises `Taken` where this monitor has no cluster
  config, or for a body it cannot read.
- `async def presets(self) -> dict[str, Any]` — Every preset's newest version, each with how many versions it has (`rollout_train.presets`).
- `async def preset(self, name: str) -> dict[str, Any] | None` — A preset's versions, oldest first; none where there is no such preset (or it was deleted).
- `async def save_preset(self, name: str, body: Mapping[str, Any]) -> dict[str, Any]` — Save `body`'s settings (`{"settings": {KEY: VALUE}, "note"}`) as a preset's next version. Raises `Taken`
  for a name that cannot be one, settings that are not a table of run settings, or a ledger that keeps no
  presets.
- `async def delete_preset(self, name: str) -> dict[str, Any]` — Delete a preset (its versions stay readable by number). Raises `KeyError` for one there is none of.
- `async def launch(self, body: Mapping[str, Any]) -> dict[str, Any]` — Ask for a run (`check`'s body), and start its job if nothing refuses it (`rollout_train.submitting
  .submit`): the launch, and the notes beside. Raises `rollout_train.launching.Refused` with the findings that
  refuse it, `Taken` where this monitor has no cluster config or the body says no name.
- `async def environments(self) -> dict[str, Any]` — Every environment the system knows of, by `module:name` (`rollout_train.monitor.environments.listed`): those
  the cluster offers, those runs were started on and those suites' versions play; each with a readable
  `name`, the versions of it seen (in runs' starts and suites' entries), whether the cluster offers it
  (`offered`), its training runs and suites, and when a run last started on it (`used`).
- `async def environment(self, environment: str) -> dict[str, Any] | None` — An environment's page (`rollout_train.monitor.environments.page_of`): what it says of itself where it loads
  in this process (its version, rows, eval data, description and curriculum; else why it does not load) and what
  the ledger has of it (each row played, its runs, suites, evals and newest check). None where it neither loads
  nor is known.
- `async def environment_versions(self) -> dict[str, Any]` — Every published environment's version the ledger keeps (`rollout_train.published`), the newest imported
  first.
- `async def environment_version(self, reference: str) -> dict[str, Any] | None` — A published environment's version, by its id or `NAME@VERSION`; none where the ledger keeps no such one.
- `async def imports(self) -> dict[str, Any]` — The imports this monitor made since it started, newest first: each with what it imports, its stage
  (`fetching`, `reading`, `packing`, `storing`, `checking`, `recording`, then `done` or `refused`), when it began
  and ended, and the version it made or why it was refused.
- `async def import_environment(self, body: Mapping[str, Any]) -> dict[str, Any]` — Import an environment from git (`rollout_train.publishing.publish`): `url`, and optionally `ref`,
  `subdirectory` and `entry_point`. Returns the version (`version`) and whether it was there already
  (`existing`). Raises `Taken` where this monitor imports nothing or the body says no URL, `Refused` saying why
  the import cannot be made.
- `async def save_suite(self, name: str, body: Mapping[str, Any]) -> Suite` — Make a suite, or its next version, as the page's forms say it (`rollout_train.evals.make_suite`,
  `edit_suite`): its `entries`, each its `environment` (`module:name`), how its starts are `chosen` (`eval data`,
  of the name `eval_data`; `rows and seeds`, `rows` (none: every row) and `seeds`; `starts`, each a row (`task`)
  and its `seed`, drawn as the row's start with that seed unless it says its `parameters`; or, for an edit,
  `same`: those of the edited version's entry of that environment), the `episodes` of each start and the limits
  (`thinking_tokens`, `answer_tokens`; null for the channel's own); and, for an edit, the version it was made from
  (`base`, by number). A body with no `entries` is one entry. Raises `Taken` for what cannot be: a name that is no
  name, no entries, an environment that does not load here or is in two entries, eval data or a row an
  environment does not have, seeds that are no whole numbers, counts below 1, an edit made from another version
  than the newest, or one that changes nothing.
- `async def stop(self, id: str) -> Launch` — Ask a launch to stop (`rollout_train.submitting.stopped`): one whose job was not made yet stops at once; a
  job going is asked to stop, and its run stops at a group boundary. Raises `KeyError` when there is no such
  launch going.
- `async def settings(self, run: str) -> dict[str, Any] | None` — A training run's settings (`rollout_train.settings`): its fixed ones and its changeable ones as its newest
  start says, what is wanted of them now, those its newest step used, and each step that used other settings than
  the one before, with what changed; and where its checkpoints go (`rollout_train.launching.checkpoints_at`, in
  the store its newest start wrote to), where this monitor has the cluster config. None where there is no such
  run.
- `async def want(self, run: str, settings: Mapping[str, Any]) -> Desired` — Want these of a run's changeable settings from its next step on. Raises `Taken` for a setting it does not
  have or cannot change, or a value it cannot take (a suite of another environment than the run's, say);
  `KeyError` where there is no such run, or nowhere to keep what is wanted. A suite the ledger does not have is
  taken: the run resolves it from its environment's eval data (`rollout_train.evals.suite_for`), or evaluates
  nothing.
- `async def checkpoint_evals(self, checkpoint: str) -> dict[str, Any] | None` — Every eval a checkpoint (by its id or the start of it) has had, by hand or by a schedule, newest first
  (`rollout_train.monitor.scores.evals_of`); None where there is no such checkpoint.
- `async def path(self, checkpoint: str) -> dict[str, Any] | None` — A checkpoint's line from the base model, with each point's scores at each suite
  (`rollout_train.monitor.scores.path_of`); None where there is no such checkpoint.
- `async def eval_subjects(self) -> dict[str, Any]` — Every subject that has had an eval, the one evaluated last first
  (`rollout_train.monitor.scores.subjects_in`).
- `async def history(self, kind: str, reference: str) -> dict[str, Any] | None` — A subject's history: every eval a checkpoint (by its id or the start of it) or a base model (by name) has had
  (`rollout_train.monitor.scores.history_of`); None where there is no such subject.
- `async def evals(self) -> dict[str, Any]` — Every suite (the version its name points to, with its environment and starts; every version; and each
  subject that played it, with the version it played and how it did at each start) and every eval (its suite, the
  version it played, its checkpoint, how far it has got), newest first (`rollout_train.monitor.scores`).
- `async def lineage(self) -> dict[str, Any]` — The checkpoints as a graph, with what trains and serves them (`rollout_train.monitor.lineage`), from every
  base model that has history and every one the cluster offers.
- `def offered_models(self) -> list[str]` — The base models the cluster offers, each once: its inference providers' models and its trainers'.
- `async def statistics(self) -> dict[str, Any]` — Every run of the ledger in figures (`rollout_train.monitor.statistics`), with each run's engines'
  throughput from its runners' heartbeats, and what the runs are called.
- `async def machines(self) -> dict[str, Any]` — Every machine that beats and the roles on it, as the heartbeats and the ledger say
  (`rollout_train.monitor.machines`): the runners and the episodes their claims hold, the sandbox pools and
  their leases, the engine hosts and how far behind what their run wants each engine is, the drivers that wait
  for what they asked Ray for, and the gateways.
- `async def queue(self) -> dict[str, Any]` — How the runs share what the cluster gives them (`rollout_train.monitor.queue`): as Kueue says, where the
  cluster config names a queue (`[kubernetes] queue`), read with the API server the config names; else as the
  runs' drivers say in their beats, with the Ray cluster's totals where this process is connected to Ray.
- `async def one_reading(self) -> AsyncGenerator[None]` — Within the block, the ledger's tables, the registry's names, the checkpoints and the beats are read once,
  whatever reads them (the hub reads every topic it watches so, once a beat).
- `def read_afresh(self) -> None` — Read the ledger again within a reading (after the monitor itself changed something).
- `async def group(self, run: str, number: int, relayed: bool = False) -> dict[str, Any] | None` — One group: what was decided (the row and its start), its stage, its episodes with what each reported,
  its step and the checkpoint it made, and its outcome.
- `def feeds(self, relayed: bool = False) -> list[dict[str, Any]]` — Every episode in the feeds of the runs' directories on this machine (and, unless `relayed`, those the
  monitors elsewhere serve), summarised, newest first.
- `async def episode(self, run_id: str, after: int = 0, relayed: bool = False) -> dict[str, Any]` — One episode: the run's lines from index `after` on (from the feed, or, once the feed has let it go, its
  replies and tool calls from the events its runner kept; where neither has a sample, as a harness's are where a
  gateway elsewhere recorded them, the replies of the turns the gateway recorded), from which its rollouts (one
  per model slot) are drawn; what it reported when it ended; and where it sits: its run, its group and its
  labels. An episode of a run on another machine is asked of the monitor there.

## `rollout_train.pods`

GPU pods elsewhere: identities, leases, a run's pods, the reaper, the trainer's client.

### `GATEWAY_IDENTITY`

*constant* · `libraries/rollout-train/src/rollout_train/pods/identity.py`

```python
GATEWAY_IDENTITY = f'spiffe://{TRUST_DOMAIN}/gateway'
```

The identity of the gateway's client certificate: the only client a pod takes requests from.

### `HELD`

*constant* · `libraries/rollout-train/src/rollout_train/pods/leases.py`

```python
HELD = 'held'
```

A lease's state while its run holds its pod.

### `IDLE`

*constant* · `libraries/rollout-train/src/rollout_train/pods/leases.py`

```python
IDLE = 'idle'
```

A lease's state once its run released its pod: no run holds it, and it is kept warm.

### `LeaseLost`

*class* · `libraries/rollout-train/src/rollout_train/pods/leasing.py`

```python
class LeaseLost(RuntimeError)
```

A lease the run held was taken from it (a reaper found it stale).

### `live`

*function* · `libraries/rollout-train/src/rollout_train/pods/identity.py`

```python
def live(beats: Iterable[Beat], role: str | None = None) -> list[LivePod]
```

The pods whose newest beat is fresh and names the identity named for the pod (of `role`, if given), by name. A
beat that says another identity than its pod's name is left out: nothing reaches it. Whatever else a beat says (an
address, say) is not read.

### `LivePod`

*class* · `libraries/rollout-train/src/rollout_train/pods/identity.py`

```python
class LivePod
```

A pod that is alive, as its newest beat says: its name, the identity its certificate must carry, its role
(`inference` or `trainer`), whether it is ready, and its certificate's serial (for whoever started the pod, which
has the certificate of a pod it no longer counts as its own revoked). Where it is reached is its lease's.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `identity` | `str` | required |  |
| `role` | `str` | required |  |
| `ready` | `bool` | required |  |
| `serial` | `str \| None` | `None` |  |

### `needs_of`

*function* · `libraries/rollout-train/src/rollout_train/pods/leasing.py`

```python
def needs_of(settings: 'RunSettings', cluster: 'Cluster') -> list[PodNeed]
```

The pods a run's settings need of the cluster's RunPod providers: one per replica of each channel on a
`runpod-inference` or `runpod-host` provider (a host's doing the trained channel's steps too, where the run's
trainer is a `runpod-trainer` on it), and one for a `runpod-trainer` of its own.

### `pod_identity`

*function* · `libraries/rollout-train/src/rollout_train/pods/identity.py`

```python
def pod_identity(name: str) -> str
```

The identity a pod's certificate carries: `spiffe://rollout/pod/NAME`. A name is lowercase letters, digits and
hyphens, at most 63 characters, as a DNS label is.

### `pod_leases_of`

*function* · `libraries/rollout-train/src/rollout_train/pods/leases.py`

```python
def pod_leases_of(ledger: Ledger) -> PodLeases | None
```

The pods' leases beside a ledger: a file beside a ledger of files, tables in a database ledger's database, the
service's for a ledger reached through it.

### `PodLease`

*class* · `libraries/rollout-train/src/rollout_train/pods/leases.py`

```python
class PodLease
```

| Field | Type | Default | Description |
|---|---|---|---|
| `pod` | `str` | required | The pod's name: the key, and its certificate's identity (`spiffe://rollout/pod/NAME`). |
| `provider` | `str` | required |  |
| `slot` | `int` | required | Its place among its provider's `max_pods`. |
| `role` | `str` | required | What it does: `inference`, `trainer`, or `host` (both). |
| `image` | `str` | required |  |
| `model` | `str` | required |  |
| `gpu` | `str` | required | The GPU type, as RunPod says it gave it (the first asked for until it says). |
| `price` | `float` | required | Dollars an hour: RunPod's `costPerHr` for it, else the provider's `price`. |
| `cloud` | `str` | `'SECURE'` |  |
| `id` | `str \| None` | `None` | RunPod's id for it, once asked for. |
| `address` | `str \| None` | `None` | Where the pod is reached, `https://IP:PORT`: its public IP and the public port its 8443/tcp is mapped to, as RunPod's API says (`rollout_runpod.Pod.address`), read by whoever leased it; none until RunPod has said. The only address the pod is reached at: nothing the pod says of itself is. |
| `run` | `str \| None` | `None` | The run that holds it (none: idle). |
| `channel` | `str \| None` | `None` | The run's channel it serves (an inference or host pod). |
| `token` | `str \| None` | `None` | The ledger service's token for the pod and the run that holds it: read by the pod itself and the platform. |
| `settings` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | What the run asks of the pod beside its channel (a trainer's implementation and settings). |
| `state` | `str` | `STARTING` |  |
| `created` | `float` | `0.0` | When it was asked for. |
| `held` | `float \| None` | `None` | Since when its run has held it. |
| `renewed` | `float \| None` | `None` | When its run last renewed it. |
| `released` | `float \| None` | `None` | When it was released (idle since). |
| `version` | `int` | `0` | Its number of changes: what a compare-and-set compares. |

**Methods**

- `def to_json(self) -> dict[str, JsonValue]`
- `@classmethod def from_json(cls, said: Mapping[str, Any]) -> 'PodLease'`
- `def shown(self) -> dict[str, JsonValue]` — What may be shown of it: everything but its token.

### `PodLeases`

*class* · `libraries/rollout-train/src/rollout_train/pods/leases.py`

```python
class PodLeases(Protocol)
```

**Methods**

- `async def now(self) -> float` — The store's clock, in seconds since the epoch.
- `async def all(self) -> list[PodLease]` — Every lease, by pod.
- `async def get(self, pod: str) -> PodLease | None`
- `async def put(self, lease: PodLease, *, expect: int | None) -> PodLease` — Write a lease if the one there has version `expect` (none: there is none, and no other lease has its
  provider's slot), as version `expect + 1` (1 for a new one); the lease written. Raises `Conflict` where it
  changed since.
- `async def delete(self, pod: str, *, expect: int) -> None` — Delete a lease that has version `expect`. Raises `Conflict` where it changed since (or is gone).
- `async def times(self, run: str | None = None) -> list[PodTime]` — The pod time charged to `run` (every run's, by default), oldest first.
- `async def charge(self, entry: PodTime) -> None` — Write a run's time on a pod, in place of what was written under its key.

### `PodNeed`

*class* · `libraries/rollout-train/src/rollout_train/pods/leasing.py`

```python
class PodNeed
```

What a run needs of one RunPod provider: `count` pods doing `role`, serving `model` (on `channel`), with the
trainer's `settings` (its implementation, model and settings) for one that takes steps.

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | required |  |
| `role` | `str` | required |  |
| `count` | `int` | required |  |
| `model` | `str` | required |  |
| `channel` | `str \| None` | `None` |  |
| `settings` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` |  |

### `Pods`

*class* · `libraries/rollout-train/src/rollout_train/pods/leasing.py`

```python
class Pods
```

A run's pods on RunPod: claimed when it starts (`claim`), renewed while it runs (`renewing`), released when it
ends (`release`). `api` gives the RunPod client of a provider (by default one with the provider's key), `ca` its
step-ca (by default the one its table says, if any). `told` hears what the run waits for.

**Methods**

- `def __init__(self, run: str, cluster: 'Cluster', ledger: Ledger, *, api: Callable[[str], 'RunPod'] | None = None, ca: Callable[[PodTable], 'StepCa | None'] | None = None, environ: Mapping[str, str] | None = None, told: Callable[[Sequence[str]], Awaitable[None]] | None = None, renew: float = RENEW, look: float = LOOK) -> None`
- `def api(self, provider: str) -> 'RunPod'`
- `async def claim(self, needs: Sequence[PodNeed]) -> list[PodLease]` — Take or start the pods `needs` say, and wait until each is ready for the run (`PodsDidNotStart` where one
  is not in time: it is deleted). Raises `ValueError` where the cluster cannot give pods (no ledger service for
  them to reach).
- `def of(self, role: str | None = None, *, channel: str | None = None, provider: str | None = None) -> list[PodLease]` — The leases it holds, of `role` (`trainer` counts a host), `channel` and `provider` where said.
- `async def renewed(self) -> float` — Stamp each lease it holds, and its time on each pod; the dollars that time cost since the last renewal.
  Raises `LeaseLost` where a lease is no longer the run's.
- `async def renewing(self, spent: Callable[[float], Awaitable[None]] | None = None) -> None` — Renew every `renew` seconds until cancelled, telling `spent` the dollars each renewal adds.
- `async def release(self) -> None` — Release every lease it holds: each pod stays warm for its provider's `idle_stop` (deleted at once where that
  is 0), its warm time charged to the run.
- `async def deleted(self, lease: PodLease, why: str, *, still: Callable[[PodLease], bool] | None = None) -> bool` — Delete a lease and its pod (`delete`), and hold it no more; whether it was deleted (false where it is no
  longer the one to delete, or could not be: said).

### `PodsDidNotStart`

*class* · `libraries/rollout-train/src/rollout_train/pods/leasing.py`

```python
class PodsDidNotStart(RuntimeError)
```

A pod did not say it was ready for its run in time: it was deleted.

### `PodTime`

*class* · `libraries/rollout-train/src/rollout_train/pods/leases.py`

```python
class PodTime
```

A run's time on a pod, charged at its price.

| Field | Type | Default | Description |
|---|---|---|---|
| `key` | `str` | required | `POD/RUN/SINCE`. |
| `pod` | `str` | required |  |
| `run` | `str` | required |  |
| `provider` | `str` | required |  |
| `gpu` | `str` | required |  |
| `price` | `float` | required |  |
| `since` | `float` | required | When the run took the pod (asked for it, for a pod started for the run). |
| `until` | `float` | required | Its newest renewal, or its release. |
| `idle` | `float` | `0.0` | Seconds it stayed warm after its release, charged to this run. |
| `released` | `bool` | `False` |  |
| `closed` | `bool` | `False` | No more time is charged to it: the pod was taken by another run, or deleted. |

**Methods**

- `@property def seconds(self) -> float`
- `@property def dollars(self) -> float`
- `def to_json(self) -> dict[str, JsonValue]`
- `@classmethod def from_json(cls, said: Mapping[str, Any]) -> 'PodTime'`

### `reap`

*function* · `libraries/rollout-train/src/rollout_train/pods/leasing.py`

```python
async def reap(cluster: 'Cluster', ledger: Ledger, *, api: Callable[[str], 'RunPod'] | None = None, ca: Callable[[PodTable], 'StepCa | None'] = _step_ca, stale: float = STALE) -> list[str]
```

Delete the pods no run holds: those of leases not renewed for `stale` seconds, of idle leases past their
provider's `idle_stop`, and those RunPod lists with the cluster's tag that no lease names. What it did, in words.

### `RemoteTrainer`

*class* · `libraries/rollout-train/src/rollout_train/pods/trainer.py`

```python
class RemoteTrainer
```

Takes steps on the training service at `address`, its files through `checkpoints`' blob store. `weights` and
`budget` are what the pod's trainer makes and can take (as the cluster says of it; `describe` asks the pod);
`objective` what it trains with (the `default` preset unless given); `changeable` the settings it takes between
steps, with their values now. The pod is reached as `connection` says
(a client certificate, the CA, and the identity the pod's certificate must carry). It is asked after a step every
`every` seconds; a step fails as `TrainerUnreachable` after `patience` seconds without an answer.

**Methods**

- `def __init__(self, address: str, checkpoints: Checkpoints, *, weights: str = 'lora', budget: Budget | None = None, objective: Objective = DEFAULT, changeable: Mapping[str, JsonValue] | None = None, connection: Connection | None = None, client: httpx.AsyncClient | None = None, every: float = 2.0, patience: float = 300.0) -> None`
- `@property def changeable(self) -> Mapping[str, JsonValue]`
- `def change(self, settings: Mapping[str, JsonValue]) -> None`
- `async def describe(self) -> dict[str, Any]` — What the pod says its trainer is: its kind, model, `weights`, `budget`, and the settings it takes between
  steps.
- `async def made(self, batch: Sequence[Item], *, seed: int, parent: Checkpoint | None, into: str) -> Made` — A step from `parent`'s files where they are, in the blob store, making the checkpoint `into` (its id): what
  the pod kept, as manifests, its state whole or (`complete` false) only what was kept with the weights so far.
- `async def state(self, into: str) -> Manifest` — The whole state of the step that made `into`, once the pod has kept it; `StateLost` where it never will (the
  pod says keeping it failed, or no longer knows the step).
- `async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step` — The step, its parent's files kept in the blob store from disk, and the files it made fetched into `into`
  once its state is all kept.
- `async def aclose(self) -> None` — Close the client it made (one it was given is its giver's).

### `STARTING`

*constant* · `libraries/rollout-train/src/rollout_train/pods/leases.py`

```python
STARTING = 'starting'
```

A lease's state while its pod is asked for and not yet ready for its run.

### `TrainerBusy`

*class* · `libraries/rollout-train/src/rollout_train/pods/trainer.py`

```python
class TrainerBusy(TrainerRefused)
```

The training pod is taking another step, which it was asked for by someone else.

### `TrainerRefused`

*class* · `libraries/rollout-train/src/rollout_train/pods/trainer.py`

```python
class TrainerRefused(StepFailed)
```

The training pod (or the proxy in front of it) refused the request.

### `TrainerUnreachable`

*class* · `libraries/rollout-train/src/rollout_train/pods/trainer.py`

```python
class TrainerUnreachable(StepFailed)
```

The training pod did not answer for as long as a step waits for it.

## `rollout_train.pki`

The certificates the platform holds, from the cluster's step-ca, published as Secrets.

### `provisioner_key`

*function* · `libraries/rollout-train/src/rollout_train/pki.py`

```python
def provisioner_key(configuration: Mapping[str, Any], password: str, name: str | None = None) -> tuple[str, dict[str, Any]]
```

The name and private key (a JWK) of step-ca's JWK provisioner `name` (by default its first), from its
configuration (`ca.json`) and the provisioner's password. Raises `ValueError` where there is none, or the password
does not open its key.

### `publish` {#rollout_trainpkipublish}

*function* · `libraries/rollout-train/src/rollout_train/pki.py`

```python
async def publish(directory: Path, password: str, url: str, namespace: str, *, secret: str = 'step-ca', tls_secret: str = 'gateway-tls', provisioner: str | None = None, api: Any = None) -> str
```

Write the root and the provisioner's key (`secret`), and a new gateway certificate (`tls_secret`), from the
step-ca whose state is in `directory` and which answers at `url`; what it did, in words.

## `rollout_train.cluster`

The cluster config: infrastructure, found, read strictly, with secrets only by name.

### `auth_problem`

*function* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
def auth_problem(where: str, auth: Auth, endpoints: Sequence[str]) -> str | None
```

Why a provider reached as `auth` at `endpoints` may not be, if it may not: with no auth, only on this
machine.

### `BlobsSection`

*class* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
class BlobsSection
```

A blob store: the cluster's default (`[blobs]`), or another, by name (`[stores.NAME]`).

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `str` | `'files'` | `files`, or `module:name` of the store. |
| `settings` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | The store's settings (a `directory` for files), none of them a credential: those come from its environment, or from the variables it names (`access_key_id_env`, `secret_access_key_env`). |
| `reader` | `Mapping[str, str]` | `field(default_factory=dict[str, str])` | The variables a read-only key is read from (`access_key_id_env`, `secret_access_key_env`), for whoever only reads it (inference pods): a named store's `reader`. |

### `BridgeSection`

*class* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
class BridgeSection
```

What a bridge's task asks for, where the cluster says more than the bridge declares (`[bridges."NAME"]`).

| Field | Type | Default | Description |
|---|---|---|---|
| `bridge` | `str` | required |  |
| `cpus` | `float \| None` | `None` |  |
| `memory_gib` | `float \| None` | `None` |  |

### `CapacitySection`

*class* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
class CapacitySection
```

The most the cluster can schedule for one run (`[capacity]`): on Kubernetes with Kueue, its queue's quota. A run
whose demand exceeds it is refused (`rollout_train.demand`). Each is unbounded where it is not said.

| Field | Type | Default | Description |
|---|---|---|---|
| `cpus` | `float \| None` | `None` |  |
| `memory_gib` | `float \| None` | `None` |  |
| `gpus` | `float \| None` | `None` |  |

### `Cluster`

*class* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
class Cluster
```

A cluster, as its config describes it. It holds no secret, only references to secrets.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | What runs record as where they ran; the Ray namespace is `rollout-NAME`. |
| `ledger` | `LedgerSection` | required |  |
| `blobs` | `BlobsSection` | `field(default_factory=BlobsSection)` |  |
| `stores` | `Mapping[str, BlobsSection]` | `field(default_factory=dict[str, BlobsSection])` | Blob stores beside the default, by name (`[stores.NAME]`): an R2 bucket that RunPod's pods reach, say. |
| `scratch` | `str` | `SCRATCH` | Node-local: checkpoints in use, fetched bases, bridge work, built Pythons. |
| `ray` | `RaySection` | `field(default_factory=RaySection)` |  |
| `kubernetes` | `KubernetesSection \| None` | `None` |  |
| `tls` | `Tls \| None` | `None` |  |
| `gateway` | `GatewaySection` | `field(default_factory=GatewaySection)` |  |
| `monitor` | `MonitorSection` | `field(default_factory=MonitorSection)` |  |
| `runners` | `RunnersSection` | `field(default_factory=RunnersSection)` |  |
| `guards` | `GuardsSection` | `field(default_factory=GuardsSection)` |  |
| `inference` | `Mapping[str, InferenceProvider]` | `field(default_factory=dict[str, InferenceProvider])` |  |
| `trainers` | `Mapping[str, TrainerProvider]` | `field(default_factory=dict[str, TrainerProvider])` |  |
| `sandboxes` | `Mapping[str, SandboxesSection]` | `field(default_factory=dict[str, SandboxesSection])` |  |
| `tools` | `Mapping[str, ToolsSection]` | `field(default_factory=dict[str, ToolsSection])` |  |
| `environments` | `Mapping[str, EnvironmentSection]` | `field(default_factory=dict[str, EnvironmentSection])` |  |
| `placement` | `Mapping[str, Mapping[str, float]]` | `field(default_factory=dict[str, Mapping[str, float]])` | Custom resources each role asks for, by role. |
| `bridges` | `Mapping[str, BridgeSection]` | `field(default_factory=dict[str, BridgeSection])` |  |
| `capacity` | `CapacitySection \| None` | `None` |  |
| `described` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue], repr=False, compare=False)` | The config as it was read (relative paths made absolute): what is handed on as JSON, and `parsed` reads back. |

**Methods**

- `@property def namespace(self) -> str`
- `def secrets(self) -> dict[str, Secret]` — Every secret the config names, by where (`inference.tinker.auth.key`).
- `def secrets_of(self, role: str = 'run') -> dict[str, Secret]` — The secrets a role reads (`SECRET_ROLES`), by where, as `secrets` says them: a run's job reads every one but
  the monitor's token; the gateway the ledger's, its keys and those of the providers it samples (a hosted API's
  key, a server's token); a monitor the ledger's, the blob stores' (but their read-only keys, which only pods
  read) and its own token; the ledger service the ledger's and the platform's token; a sandbox pool the ledger's;
  the reaper the ledger's and RunPod's keys. What each role is given on Kubernetes is this
  (docs/deploy/helm.md).

### `ClusterError`

*class* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
class ClusterError(ValueError)
```

A cluster config that cannot be used, and why.

### `EnvironmentSection`

*class* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
class EnvironmentSection
```

The Python an environment runs in (`[environments."NAME"]`): the platform's, or a uv project's lock built into a
cached virtualenv.

| Field | Type | Default | Description |
|---|---|---|---|
| `environment` | `str` | required |  |
| `python` | `str \| None` | `'platform'` |  |
| `project` | `str \| None` | `None` | A uv project's directory (relative paths are from the config file's). |
| `interpreter` | `str \| None` | `None` | The interpreter a run on it starts in, where it is not the platform's: by default a project's `PROJECT/.venv/bin/python`. |

**Methods**

- `@property def runs_in(self) -> str | None` — The interpreter a run on it starts in; none: the platform's.

### `find`

*function* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
def find(given: str | None = None, environ: Mapping[str, str] | None = None) -> Path
```

The cluster config's file: `given` (`--cluster`: a path, or a name under `~/.config/rollout/clusters`), else
`ROLLOUT_CLUSTER` (the same), else `~/.config/rollout/cluster.toml`. Raises `ClusterError` where the file it
names is not there, saying where it looked.

### `GatewaySection`

*class* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
class GatewaySection
```

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | `'http://127.0.0.1:8830'` | How runners and harnesses reach it. |
| `listen` | `str` | `'127.0.0.1:8830'` | `host:port` a replica serves on. |
| `replicas` | `int` | `1` |  |
| `keys` | `Secret \| None` | `None` | The secrets keys are signed with (`keys_file`, `keys_env`). |
| `lifetime` | `float` | `21600.0` | Seconds a key minted for a slot is good for. |

### `GuardsSection`

*class* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
class GuardsSection
```

| Field | Type | Default | Description |
|---|---|---|---|
| `runs_gib` | `float` | `0.0` | System memory a node must have free before a runner claims an episode. |
| `training_gib` | `float` | `0.0` | And before a colocated step starts. |

### `inspect`

*function* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
def inspect(cluster: Cluster, environ: Mapping[str, str] | None = None, role: str = 'run') -> list[str]
```

What is wrong with the cluster on this node, in words: each secret reference `role` reads
(`Cluster.secrets_of`) that does not resolve (by name, never by value), and each environment's project with no
`uv.lock`. Empty: nothing.

### `KubernetesSection`

*class* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
class KubernetesSection
```

Where each run's job is a RayJob with a Ray cluster of its own (`[kubernetes]`): the namespace RayJobs are made
in, the template each is made from (a RayJob's YAML: its Ray cluster, image, volumes and retries; relative paths
are from the config file's), and the API server (by default the one of the cluster the process runs in, reached
with its service account).

| Field | Type | Default | Description |
|---|---|---|---|
| `namespace` | `str` | required |  |
| `rayjob` | `str` | required |  |
| `api` | `str` | `'https://kubernetes.default.svc'` |  |
| `queue` | `str \| None` | `None` | The Kueue LocalQueue in `namespace` that admits runs' RayJobs: each is made suspended, with the label `kueue.x-k8s.io/queue-name`, and starts when Kueue admits it whole. None: each starts when it is made. |

### `LedgerSection`

*class* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
class LedgerSection
```

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | `str \| None` | `None` | `sqlite:///…` on one machine, `postgresql://…` for several (with no password: that is `url_env`'s), or the ledger service's `http(s)://…` (`rollout_train.ledger_service.HttpLedger`, with `token`). |
| `url_secret` | `Secret \| None` | `None` | The URL, named, where it holds a password (`url_env`, `url_file`). |
| `token` | `Secret \| None` | `None` | The platform's token for the ledger service (`token_env`, `token_file`): what its roles send when `url` is the service's, what the service takes as the platform's, and what pods' tokens are signed with. |
| `public` | `str \| None` | `None` | Where processes outside the cluster (RunPod's pods) reach the ledger service: `https://…`. |

### `load`

*function* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
def load(path: Path) -> Cluster
```

The cluster a config file describes, checked. Raises `ClusterError` saying what is wrong and where.

### `located`

*function* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
def located(given: str | None = None, environ: Mapping[str, str] | None = None) -> Cluster
```

The cluster config this process works with: the one its job was handed (`ROLLOUT_CLUSTER_JSON`), else the file
`find` finds, read and checked. Raises `ClusterError` saying what is wrong.

### `MonitorSection`

*class* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
class MonitorSection
```

| Field | Type | Default | Description |
|---|---|---|---|
| `listen` | `str` | `'127.0.0.1:8765'` |  |
| `feed_episodes` | `int` | `80` | Episodes kept in a run's live feed. |
| `token` | `Secret \| None` | `None` | The monitor's token (`token_env`, `token_file`), which its page and every client of its API present (`rollout_train.monitor.access`); none: `ROLLOUT_MONITOR_TOKEN`. |

### `parsed` {#rollout_trainclusterparsed}

*function* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
def parsed(described: Mapping[str, Any], *, relative_to: Path | None = None) -> Cluster
```

The cluster a config's table describes, checked; relative paths of environments' projects are from
`relative_to`.

### `RaySection`

*class* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
class RaySection
```

| Field | Type | Default | Description |
|---|---|---|---|
| `address` | `str` | `'auto'` | The Ray cluster's address (its GCS, `host:port`), which a run's driver joins; `auto`: the one Ray finds (in a job, the cluster the job runs on). |
| `jobs` | `str` | `'http://127.0.0.1:8265'` | The job server. |
| `temp_dir` | `str` | `'~/.cache/ray'` | On disk: /tmp may be memory. |
| `memory_threshold` | `float` | `0.85` | Ray's memory monitor kills a task past this share of the machine's memory. |
| `python` | `str` | `'platform'` | The interpreter a run's job starts in: `platform`, the `python` on the job's `PATH` (the platform's), or a path. |

### `RunnersSection`

*class* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
class RunnersSection
```

| Field | Type | Default | Description |
|---|---|---|---|
| `places` | `int` | `8` | Episodes one runner plays at once. |

### `SandboxesSection`

*class* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
class SandboxesSection
```

A pool of sandboxes of one kind, which environments declare they need (`[sandboxes.KIND]`): made in each run's
driver from its provider, or, with `url`, served elsewhere (`rollout pool --kind KIND`), where runs reach it. What
its sandboxes run and hold is the pool's business: a run's demand counts none of it.

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `str` | required |  |
| `provider` | `str \| None` | `None` | `module:name` of what makes them. |
| `python` | `str` | `'platform'` | `platform`, or the name of an environment whose Python the provider is in. |
| `size` | `int` | `1` |  |
| `url` | `str \| None` | `None` | Where the pool is served (`rollout.harness.remote.serve_pool`): runs acquire from it there. |
| `pools` | `int` | `1` |  |
| `settings` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | The provider's own settings. |

### `ToolsSection`

*class* · `libraries/rollout-train/src/rollout_train/cluster.py`

```python
class ToolsSection
```

A tool set served elsewhere, by name (`[tools.NAME]`).

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `url` | `str` | required |  |
| `auth` | `Auth` | `field(default_factory=Auth)` |  |

## `rollout_train.providers`

Inference providers and trainers: kinds, capabilities, auth, allocation, routing.

### `ALLOCATIONS`

*constant* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
ALLOCATIONS: tuple[Allocation, ...] = ('metered', 'scheduled')
```

Metered (bounded by spend, rate limits and a concurrency cap) or scheduled (capacity a run is placed on).

### `Auth`

*class* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
class Auth
```

How a provider is reached. `token` for `bearer`; `key` for `vendor` (what the vendor's SDK reads, named so that
`rollout cluster check` can say whether it resolves); `trust` says whose CAs verify the server (`system` or
`cluster`; by default the cluster's for `mtls`, the system's otherwise); `identity` is the SPIFFE identity the
server's certificate must carry (`leased`: each server's own, named for the pod its lease names), in place of its
host name.

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `AuthKind` | `'none'` |  |
| `token` | `Secret \| None` | `None` |  |
| `key` | `Secret \| None` | `None` |  |
| `trust` | `Literal['system', 'cluster'] \| None` | `None` |  |
| `identity` | `str \| None` | `None` |  |

**Methods**

- `@property def trusts(self) -> Literal['system', 'cluster']` — Whose CAs verify the server.
- `def connection(self, tls: Tls | None = None, *, identity: str | None = None) -> Connection` — The settings a client reaches the provider's servers with: the cluster's CA or the system's; a client
  certificate only for `mtls`; the server's SPIFFE identity checked in place of its host name where one is said
  (`identity`, the pod's own, named for it, when `self.identity` is `leased`); else its host name. Raises
  `ValueError` where the cluster has no `[tls]` it needs.

### `AUTHS`

*constant* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
AUTHS: tuple[AuthKind, ...] = ('mtls', 'bearer', 'vendor', 'none')
```

How a provider is reached: mutual TLS with the cluster's CA, a bearer token, the vendor's SDK, or nothing (only on
this machine).

### `Capabilities`

*class* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
class Capabilities
```

What an inference provider's kind can do.

| Field | Type | Default | Description |
|---|---|---|---|
| `token_exact` | `bool` | required | Takes token ids and returns the exact sampled ids. |
| `sampled_logprobs` | `bool` | required | Each sampled token's logprob, under the distribution it was sampled from. |
| `prompt_logprobs` | `bool` | required | Logprobs of given tokens (scoring without sampling). |
| `top_logprobs` | `int` | required | How many of the top logprobs it can return per position (0: none). |
| `honours_sampling` | `bool` | required | Temperature and top-p are applied, so recorded logprobs describe what was sampled. |
| `adapters` | `bool` | required | Serves LoRA adapters by name. |
| `full_reload` | `bool` | required | Serves new full weights under a checkpoint's name. |
| `streaming` | `bool` | required | Replies as a stream. |
| `loads` | `frozenset[str]` | required | The checkpoint formats it serves (`peft`, `full`, `tinker`); none: base models only. |
| `bills` | `Literal['none', 'tokens', 'hours']` | `'none'` | What it costs by: nothing, tokens (per model and token class), or hours of pods. |
| `unchecked` | `frozenset[str]` | `frozenset()` | Capabilities declared as the SDK says but not yet confirmed by a live test: nothing relies on them until then (Tinker's prompt and top-k logprobs). |

**Methods**

- `@property def sampled_with(self) -> tuple[str, ...]` — What its turns are sampled with, as a turn records it (`rollout_train.recorder.segments.TOKEN_LEVEL`).

### `INFERENCE_KINDS`

*constant* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
INFERENCE_KINDS: Mapping[str, InferenceKind] = {each.name: each for each in (InferenceKind('vllm', _token_level(prompt_logprobs=True, top_logprobs=20, full_reload=True, loads=frozenset({'peft', 'full'})), auths=('none', 'bearer', 'mtls'), auth=Auth('none'), fields=('engine', 'listen', 'max_logprobs'), implementation='rollout_vllm:VllmEngine'), InferenceKind('vllm-servers', _token_level(prompt_logprobs=True, top_logprobs=20, full_reload=False, loads=frozenset({'peft'})), auths=('none', 'bearer', 'mtls'), auth=None, fields=('addresses', 'via', 'loader', 'max_logprobs'), implementation='rollout_train.inference:RemoteEngine', remote=True), InferenceKind('tinker', _token_level(prompt_logprobs=True, top_logprobs=20, full_reload=False, loads=frozenset({'tinker'}), bills='tokens', unchecked=frozenset({'prompt_logprobs', 'top_logprobs'})), auths=('vendor',), auth=Auth('vendor', key=Secret(env='TINKER_API_KEY')), fields=('project',), secrets=('project',), implementation='rollout_tinker:TinkerEngine', allocation='metered'), InferenceKind('api', Capabilities(token_exact=False, sampled_logprobs=False, prompt_logprobs=False, top_logprobs=0, honours_sampling=False, adapters=False, full_reload=False, streaming=True, loads=frozenset(), bills='tokens'), auths=('vendor', 'bearer'), auth=Auth('vendor'), fields=('endpoint', 'base_url'), secrets=('api_key',), allocation='metered'), InferenceKind('runpod-inference', _token_level(prompt_logprobs=True, top_logprobs=20, full_reload=False, loads=frozenset({'peft'}), bills='hours'), auths=('mtls',), auth=Auth('mtls', identity=LEASED), fields=(*POD_FIELDS, 'max_logprobs', 'memory_fraction'), secrets=('api_key',), implementation='rollout_train.pods.inference:InferencePod', remote=True), InferenceKind('runpod-host', _token_level(prompt_logprobs=True, top_logprobs=20, full_reload=False, loads=frozenset({'peft'}), bills='hours'), auths=('mtls',), auth=Auth('mtls', identity=LEASED), fields=(*POD_FIELDS, 'max_logprobs', 'memory_fraction', 'sleep'), secrets=('api_key',), implementation='rollout_train.pods.inference:InferencePod', remote=True))}
```

Every kind of inference provider, by name.

### `InferenceKind`

*class* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
class InferenceKind
```

What a kind of inference provider is, whatever cluster it is in.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `capabilities` | `Capabilities` | required |  |
| `auths` | `tuple[AuthKind, ...]` | required | The ways it may be reached. |
| `auth` | `Auth \| None` | required | How it is reached when the cluster config does not say (none: the config must say). |
| `fields` | `tuple[str, ...]` | required | The settings of its `[inference.NAME]` table beyond those every provider has. |
| `secrets` | `tuple[str, ...]` | `()` | The secrets its table may name (`NAME_env`, `NAME_file`), beyond its auth's. |
| `implementation` | `str \| None` | `None` | `module:name` of what samples it, where one module does. |
| `remote` | `bool` | `False` | Whether its servers are reached at addresses (and so need an auth other than `none` unless local). |
| `allocation` | `Allocation` | `'scheduled'` | Metered or scheduled, when the cluster config does not say. |

### `InferenceProvider`

*class* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
class InferenceProvider
```

An inference provider as a cluster deploys it (`[inference.NAME]`).

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `kind` | `str` | required |  |
| `capabilities` | `Capabilities` | required | The kind's, with this deployment's `max_logprobs` as its top-k logprobs: what a `vllm` provider's engines are started with, and what a `vllm-servers` or `runpod-inference` provider's servers were (`--max-logprobs`). |
| `models` | `Mapping[str, ModelOffer]` | required |  |
| `auth` | `Auth` | required |  |
| `gpus` | `float` | `0` | Per replica. |
| `replicas` | `int` | `1` | Per run channel, unless the run asks for more (`channels.NAME.replicas`). |
| `endpoints` | `tuple[str, ...]` | `()` | Where its servers are reached (addresses, a router, where local engines listen). |
| `settings` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | The rest of its table: its kind's own settings, none of them a secret. |
| `secrets` | `Mapping[str, Secret]` | `field(default_factory=dict[str, Secret])` | The secrets its table names, beyond its auth's (RunPod's API key, Tinker's project). |
| `allocation` | `Allocation` | `'scheduled'` |  |
| `concurrency` | `int \| None` | `None` | For a metered provider, the most requests it is sent at once (none: as many as runs send). |

**Methods**

- `@property def local(self) -> bool` — Whether every endpoint it is reached at is on this machine.

### `is_local`

*function* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
def is_local(address: str) -> bool
```

Whether an address (a URL, or a host) is on this machine: `localhost` or a loopback address.

### `ModelOffer`

*class* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
class ModelOffer
```

A model a provider serves here.

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required |  |
| `context` | `int` | required | The longest sequence it takes, prompt and reply. |
| `base` | `str \| None` | `None` | The model it was quantized from, if any: an adapter trained over the base can be served on it. |
| `max_lora_rank` | `int \| None` | `None` | The highest adapter rank it loads (none: adapters are not limited here, or not served). |
| `cost` | `Mapping[str, float]` | `field(default_factory=dict[str, float])` | Dollars per million tokens by token class (`input`, `cached_input`, `output`, `thinking`), or per hour (`hour`). Cached input is priced as input, and thinking as output, where the table does not say. |
| `options` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | What its engines are started with (`gpu_memory_utilization`, `max_num_seqs`, …); for a hosted API's model, what it takes (`max_output_tokens`, and its endpoint's own: how it thinks, whether it takes sampling). |

### `POD_FIELDS`

*constant* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
POD_FIELDS = ('image', 'gpu_types', 'gpu_count', 'max_pods', 'idle_stop', 'start_timeout', 'cloud', 'regions', 'cuda_versions', 'price', 'volume_gb', 'container_disk_gb', 'secrets', 'step_ca', 'store')
```

The settings of a RunPod kind's table that say what its pods are (`PodTable`).

### `pod_table`

*function* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
def pod_table(kind: str, settings: Mapping[str, JsonValue]) -> PodTable
```

What a RunPod kind's table says of its pods, checked. Raises `ValueError` saying what is wrong.

### `PodTable`

*class* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
class PodTable
```

What a RunPod kind's table says of its pods (`pod_table`).

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | `str` | required | The image its pods run (a digest from the images workflow's summary). |
| `gpu_types` | `tuple[str, ...]` | required | RunPod's GPU type ids, in order of preference (`NVIDIA H100 80GB HBM3`). |
| `gpu_count` | `int` | `1` | GPUs a pod has (a host's: one). A trainer's pod of more steps on all of them, the policy sharded over them. |
| `max_pods` | `int` | `1` | The most pods of the provider at once, across runs: a cap on what it spends. |
| `idle_stop` | `float` | `600.0` | Seconds a pod no run holds stays warm, for the next run with the same image, model and GPU, before it is deleted. |
| `start_timeout` | `float` | `1200.0` | Seconds a pod may take to say it is ready for the run that holds it before it is deleted and the run fails. |
| `cloud` | `str` | `'SECURE'` | RunPod's cloud tier: `SECURE` or `COMMUNITY`. |
| `regions` | `tuple[str, ...]` | `()` | RunPod's data centers its pods may be in (none: any). |
| `cuda_versions` | `tuple[str, ...]` | `('13.0',)` | The CUDA versions a pod's machine may support (RunPod's `allowedCudaVersions`): its NVIDIA driver must run what the image was built for. The pods' images are built on CUDA 13, which an older driver refuses. |
| `price` | `float \| None` | `None` | Dollars an hour a pod is reckoned at before RunPod says its own (`costPerHr`): what estimates use. |
| `volume_gb` | `int` | `50` |  |
| `container_disk_gb` | `int` | `50` |  |
| `secrets` | `Mapping[str, str]` | `field(default_factory=dict[str, str])` | Variables whose values are RunPod console secrets, by the secret's name (`HF_TOKEN = "hf_token"`). |
| `step_ca` | `Mapping[str, str]` | `field(default_factory=dict[str, str])` | step-ca, for the pods' certificates: `url`, `provisioner`, `key_file` (the provisioner's key), `root` (the cluster's root, which pods are given), and `trust` (`root`, by default: pods reach step-ca directly and check its TLS by the root; `system`: behind a proxy that ends TLS with a public certificate, they check it by the system's roots, and renew with a token rather than over mutual TLS). |
| `store` | `str \| None` | `None` | The blob store its pods read and write (`[stores.NAME]`; none: `[blobs]`). |
| `memory_fraction` | `float \| None` | `None` | The share of the GPU's memory vLLM takes (`--gpu-memory-utilization`); on a `runpod-host` pod the trainer has the rest (0.42 unless said). |
| `sleep` | `bool` | `False` | On a `runpod-host` pod: whether vLLM sleeps while a step is taken (`rollout_train.colocated`), for a GPU too small to hold both. |

### `ROUTING`

*constant* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
ROUTING = ('spill', 'weighted')
```

### `Routing`

*class* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
class Routing
```

How a channel served by several providers shares its turns among them.

| Field | Type | Default | Description |
|---|---|---|---|
| `rule` | `Literal['spill', 'weighted']` | `'spill'` |  |
| `providers` | `tuple[str, ...]` | `()` | In order: for `spill`, the first is filled before the next is asked. |
| `weights` | `Mapping[str, float]` | `field(default_factory=dict[str, float])` | For `weighted`: each provider's weight. |

### `RUNPOD`

*constant* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
RUNPOD = ('runpod-inference', 'runpod-host', 'runpod-trainer')
```

The kinds whose servers are RunPod's pods, leased by the runs that use them.

### `Secret`

*class* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
class Secret
```

A secret, named: an environment variable (`env`) or a file (`file`). The value is read where it is used, at the
moment it is needed (`resolve`), and never kept.

| Field | Type | Default | Description |
|---|---|---|---|
| `env` | `str \| None` | `None` |  |
| `file` | `str \| None` | `None` |  |

**Methods**

- `def resolve(self, environ: Mapping[str, str] | None = None) -> str | None` — The secret's value, read now (none where the variable is unset or empty, or the file is missing).

### `settings_of`

*function* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
def settings_of(kind: TrainerKind) -> tuple[SettingSpec, ...]
```

The settings a kind of trainer takes, by `trainer.FIELD`, with their types, defaults and whether they are
changeable: read from its settings dataclass and the `CHANGEABLE` beside it (importing neither the trainer nor
torch). Raises `ImportError` where the trainer's package is not installed here.

### `SettingSpec`

*class* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
class SettingSpec
```

One setting a trainer takes, read from its settings dataclass.

| Field | Type | Default | Description |
|---|---|---|---|
| `key` | `str` | required | As a run names it: `trainer.FIELD`. |
| `types` | `tuple[str, ...]` | required | The JSON types it takes: `int`, `float`, `bool`, `str`, `null`. |
| `default` | `JsonValue` | required |  |
| `changeable` | `bool` | required | Whether a running run takes it from its next step on. |

**Methods**

- `def accepts(self, value: JsonValue) -> bool` — Whether `value` is of a type it takes (a whole number is a float too; a bool is neither).

### `Tls`

*class* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
class Tls
```

The cluster's own certificate authority and the client certificate its gateway and runs' drivers present
(the cluster config's `[tls]`): paths, never the keys themselves.

| Field | Type | Default | Description |
|---|---|---|---|
| `ca` | `str \| None` | `None` | The cluster CA's root certificate. |
| `certificate` | `str \| None` | `None` |  |
| `key` | `str \| None` | `None` | The client certificate and its key's file, for providers reached over mutual TLS. |
| `identity` | `str` | `'spiffe://rollout/gateway'` | The SPIFFE identity the client certificate carries. |

### `TRAINER_KINDS`

*constant* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
TRAINER_KINDS: Mapping[str, TrainerKind] = {each.name: each for each in (TrainerKind('lora', _LORA, 'rollout_lora:LoraTrainer', 'rollout_lora.settings:LoraSettings', auths=('none',), auth=Auth('none'), fields=('implementation', 'gpu_memory_gib'), not_settings={**_OBJECTIVE, 'frozen_reference': "an adapter's reference is the model with the adapter switched off"}), TrainerKind('full', _FULL, 'rollout_lora:FullTrainer', 'rollout_lora.settings:LoraSettings', auths=('none',), auth=Auth('none'), fields=('implementation', 'gpu_memory_gib'), not_settings={**_OBJECTIVE, 'rank': 'a full-weight trainer has no adapter', 'whole_base': 'every weight is trained, and sharded'}), TrainerKind('tinker', TrainerCapabilities('lora', 'tinker', _EVERY_FAMILY, True, frozenset({'tinker'}), reference='no', entropy=False, distribution=False), 'rollout_tinker:TinkerTrainer', 'rollout_tinker.settings:TinkerSettings', auths=('vendor',), auth=Auth('vendor', key=Secret(env='TINKER_API_KEY')), fields=('project', 'implementation'), secrets=('project',), not_settings={**_OBJECTIVE, 'project': 'the cluster config says it ([trainers.NAME] project)'}, allocation='metered'), TrainerKind('runpod-trainer', _LORA, 'rollout_train.pods:RemoteTrainer', 'rollout_lora.settings:LoraSettings', auths=('mtls',), auth=Auth('mtls', identity=LEASED), fields=('trainer', 'gpu_memory_gib', *POD_FIELDS), secrets=('api_key',), not_settings=_OBJECTIVE))}
```

Every kind of trainer, by name.

### `TrainerCapabilities`

*class* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
class TrainerCapabilities
```

What a kind of trainer makes and takes.

| Field | Type | Default | Description |
|---|---|---|---|
| `produces` | `Literal['lora', 'full']` | required |  |
| `format` | `str` | required | The checkpoint format its files are in: `peft`, `full` or `tinker`. |
| `families` | `frozenset[str]` | required | The objective families it takes (`rollout_train.objectives.FAMILIES`). |
| `scores` | `bool` | required | Can compute logprobs of given tokens (for distillation, and supervised data without behaviour logprobs). |
| `starts_from` | `frozenset[str]` | required | Checkpoint formats a run may start from (besides the base model). |
| `reference` | `Literal['yes', 'asked', 'no']` | `'yes'` | Whether it gives the reference model's logprobs, which a KL to the reference and most preference losses read: `yes` (an adapter switched off), `asked` (only when its settings ask, `trainer.frozen_reference`: a frozen copy of the model beside the policy), `no`. |
| `entropy` | `bool` | `True` | Whether it gives each position's entropy (an entropy bonus reads it). |
| `distribution` | `bool` | `True` | Whether it gives the policy's logprobs of tokens other than the sampled ones (the top-k form of distillation reads them at the teacher's top-k tokens). |

### `TrainerKind`

*class* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
class TrainerKind
```

What a kind of trainer is, whatever cluster it is in.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `capabilities` | `TrainerCapabilities` | required |  |
| `implementation` | `str` | required | `module:name` of the trainer. |
| `settings` | `str` | required | `module:name` of its settings dataclass: each field a setting `trainer.FIELD`, the module's `CHANGEABLE` the ones it takes between steps. |
| `auths` | `tuple[AuthKind, ...]` | required |  |
| `auth` | `Auth` | required |  |
| `fields` | `tuple[str, ...]` | `()` | The settings of its `[trainers.NAME]` table beyond those every trainer has (`implementation`: what makes the trainer, `module:name`, in place of the kind's own, called as it is). |
| `secrets` | `tuple[str, ...]` | `()` |  |
| `not_settings` | `Mapping[str, str]` | `field(default_factory=dict[str, str])` | Fields of its settings dataclass a run does not set, and why. |
| `allocation` | `Allocation` | `'scheduled'` | Metered or scheduled, when the cluster config does not say. |

### `TrainerProvider`

*class* · `libraries/rollout-train/src/rollout_train/providers.py`

```python
class TrainerProvider
```

A trainer as a cluster deploys it (`[trainers.NAME]`).

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `kind` | `str` | required |  |
| `capabilities` | `TrainerCapabilities` | required |  |
| `models` | `tuple[str, ...]` | required | The models it trains here. |
| `auth` | `Auth` | required |  |
| `segment_tokens` | `int \| None` | `None` | The longest segment this hardware trains on (none: any). |
| `gpus` | `float` | `0` | The GPUs it steps on (a share of one, where it shares an engine's): above one, a whole number, which it steps on together, the policy sharded over them. A `runpod-trainer`'s are its pods' `gpu_count`. |
| `colocate_with` | `str \| None` | `None` | A `vllm` provider whose GPU it shares (that provider's engines sleep while it steps), or, for a `runpod-trainer`, a `runpod-host` provider whose pods take its steps beside their vLLM. |
| `cost` | `Mapping[str, float]` | `field(default_factory=dict[str, float])` | Dollars per million tokens trained (`train`), or per hour (`hour`), for a model `costs` does not name. |
| `costs` | `Mapping[str, Mapping[str, float]]` | `field(default_factory=dict[str, Mapping[str, float]])` | Its cost for each model whose price differs, by model, in the units of `cost`. |
| `settings` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | The rest of its table: its kind's own settings, none of them a secret. |
| `secrets` | `Mapping[str, Secret]` | `field(default_factory=dict[str, Secret])` |  |
| `allocation` | `Allocation` | `'scheduled'` |  |
| `concurrency` | `int \| None` | `None` | For a metered trainer, the most requests it is sent at once (none: as many as the run sends). |

**Methods**

- `def cost_of(self, model: str) -> Mapping[str, float]` — What training `model` here costs: its own entry in `costs`, else `cost`.
- `@property def runs(self) -> TrainerKind` — The kind of trainer its steps are taken by (for `runpod-trainer`, the one it names).
- `@property def implementation(self) -> str` — `module:name` of what makes the trainer: its table's `implementation`, else its kind's.

## `rollout_train.bridges`

Bridges between checkpoint formats: the registry, paths, refused pairs, their tasks.

### `Bridge`

*class* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
class Bridge
```

One bridge: from a format to another, the task that does it, and what the task needs.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `source` | `str` | required |  |
| `target` | `str` | required |  |
| `task` | `str \| None` | required | `module:name` of the function that writes the target's files from the source's; none: the files are served as they are, with nothing written. |
| `says` | `str` | required |  |
| `cpus` | `float` | `1` |  |
| `memory_gib` | `float` | `1` |  |
| `network` | `bool` | `False` | Whether the task fetches from outside the cluster (Tinker's archive). |
| `rank_factors` | `tuple[tuple[str, int], ...]` | `()` | By model pattern (`fnmatch`): how many times the trained rank the provider sees. |
| `explicit` | `bool` | `False` | Chosen only when the run asks for it (`channels.NAME.bridge`), never by the path search alone. |
| `cost` | `int` | `1` | Its weight in the path search. |

### `bridge_of`

*function* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
async def bridge_of(ledger: Ledger, checkpoint: str) -> str | None
```

The bridge, by name, that last made files of a checkpoint, if one did.

### `BRIDGED`

*constant* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
BRIDGED = 'checkpoints/resharded'
```

The ledger's table of what each bridge made, keyed `CHECKPOINT@BRIDGE`.

### `bridged`

*function* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
async def bridged(checkpoints: Checkpoints, fence: Fence, checkpoint: str, chain: Sequence[Bridge], scratch: Path, *, target: str | None = None, settings: Mapping[str, Mapping[str, JsonValue]] | None = None) -> Manifest
```

The files a provider loads for `checkpoint` (by id), made in this process by each bridge of `chain` (`path`'s)
in turn, each from what the one before made, the first from the checkpoint's weights, unless it made them before.
A bridge with no task (`none`) passes on the files it is given. `scratch` is where the files are read to and
written, on this machine, for the while it takes; `target` is the model the provider serves, and `settings` each
bridge's own, by its name.

### `BRIDGES`

*constant* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
BRIDGES: tuple[Bridge, ...] = (Bridge('none', 'tinker', 'tinker', None, "served as it is: Tinker's sampler reads the checkpoint's pointer", cost=0), Bridge('peft-from-tinker', 'tinker', 'peft', 'rollout_tinker.bridges:peft', "Tinker's adapter downloaded and written in PEFT's layout", cpus=2, network=True, rank_factors=(('Qwen/Qwen3.5-*', 3),)), Bridge('verbatim', 'peft', 'peft', VERBATIM, "the adapter's files, linked as they are", cpus=0.5, memory_gib=1), Bridge('full-reload', 'full', 'full', VERBATIM, "the full weights' files, linked as they are and loaded under the checkpoint's name, replica by replica", cpus=0.5, memory_gib=1), Bridge(MERGE_QUANTIZE, 'peft', 'full', 'rollout_lora.bridges:merge_quantize', 'the adapter merged into its base, as full weights a provider quantizes as it loads them', cpus=8, memory_gib=48, explicit=True, cost=10))
```

Every bridge, by its pair of formats.

### `BRIDGING`

*constant* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
BRIDGING = 'checkpoints/resharding'
```

The ledger's table of bridges begun, keyed `CHECKPOINT@BRIDGE` (`key`).

### `by_name`

*function* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
def by_name(name: str) -> Bridge
```

The bridge of the registry called `name`; `KeyError`, saying the names, for none.

### `checkpoint_of`

*function* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
def checkpoint_of(entry: str) -> str
```

The checkpoint an entry of the bridges' tables is of (`key`).

### `Context`

*class* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
class Context
```

What a bridge's task is told beside the files it reads and writes.

| Field | Type | Default | Description |
|---|---|---|---|
| `checkpoint` | `str` | required | The checkpoint it bridges, by id. |
| `model` | `str \| None` | `None` | The model the checkpoint's weights are over (its record's `base`), by name or directory. |
| `target` | `str \| None` | `None` | The model the provider serves it on (a copy of `model` quantized, say); none: `model`. |
| `settings` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | The bridge's own settings, from whoever runs it (the service Tinker's bridge asks, say). |

### `format_of`

*function* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
def format_of(files: Collection[str]) -> frozenset[str]
```

The formats a checkpoint's files are in, by their paths within it (a manifest's, with or without the leading
`weights/`): `tinker.json` is `tinker`; `adapter_config.json` with `adapter_model.safetensors` is `peft`;
`config.json` with safetensors weights is `full`. A checkpoint can be in two at once.

### `FORMATS`

*constant* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
FORMATS = ('peft', 'full', 'tinker')
```

The checkpoint formats: an adapter in PEFT's layout, full weights (safetensors and a `config.json`), and pointers
to Tinker's sampler checkpoint and state.

### `key`

*function* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
def key(checkpoint: str, bridge: str) -> str
```

A checkpoint's entry in the bridges' tables, for a bridge by name: `CHECKPOINT@BRIDGE`.

### `made`

*function* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
async def made(ledger: Ledger, checkpoint: str, bridge: str) -> Manifest | None
```

What a bridge, by name, made of a checkpoint, if it did.

### `NoBridge`

*class* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
class NoBridge
```

A pair of formats with no path between them, and why.

| Field | Type | Default | Description |
|---|---|---|---|
| `source` | `str` | required |  |
| `target` | `str` | required |  |
| `reason` | `str` | required |  |

### `on_ray`

*function* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
async def on_ray(ledger_at: Mapping[str, Any], blobs_at: Mapping[str, Any], fence: Fence, checkpoint: str, chain: Sequence[Bridge], *, target: str | None = None, settings: Mapping[str, Mapping[str, JsonValue]] | None = None, scratch: str = SCRATCH, placement: Mapping[str, Any] | None = None) -> Manifest
```

`bridged`, with each bridge of `chain` a Ray task of its own on the cluster this process is connected to
(`ray.init`), asking for the CPUs and memory the bridge declares, or those its `settings` say (`cpus`,
`memory_gib`: the cluster's `[bridges."NAME"]`). `placement` places each task (a run's: in the bridge's bundle of
its placement group, `rollout_train.demand.placed`). `ledger_at` and `blobs_at` say where a worker finds the ledger
and the blob store; `scratch` is where it works, on its own machine. A chain whose bridges write nothing (`none`)
serves the checkpoint's own files.

### `path`

*function* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
def path(source: str, loads: Collection[str], *, wanted: str = AUTO) -> tuple[Bridge, ...] | NoBridge
```

The cheapest chain of bridges from `source` to one of the formats in `loads`. `wanted` is the run's
`channels.NAME.bridge`: `auto`, or `merge-quantize` for a path through it (a bridge marked `explicit` is used
only then).

### `rank_factor`

*function* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
def rank_factor(chain: Collection[Bridge], model: str) -> int
```

How many times the trained rank a provider sees, after the bridges of `chain`, for an adapter over `model`.

### `REFUSED`

*constant* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
REFUSED: tuple[NoBridge, ...] = (NoBridge('peft', 'tinker', _NO_UPLOAD), NoBridge('full', 'tinker', _NO_UPLOAD), NoBridge('full', 'peft', 'full weights are not an adapter: serve them on a provider that reloads full weights'))
```

Pairs refused on purpose, with the reason a run is told.

### `verbatim`

*function* · `libraries/rollout-train/src/rollout_train/bridges.py`

```python
def verbatim(weights: Path, into: Path, context: Context) -> dict[str, JsonValue]
```

The provider loads the trainer's files as they are: each is linked (or copied) into `into`. A chain notes what
this bridge makes as the checkpoint's own manifest without running it (the same files are the same blobs); it is the
task for a caller that has the files on disk.

## `rollout_train.objectives`

Objectives declared: families, components, presets, and resolving them.

### `Advantage`

*class* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
class Advantage
```

How a policy-gradient segment's advantage is made from its group's scores (`rollout_train.algorithm`).

| Field | Type | Default | Description |
|---|---|---|---|
| `baseline` | `str` | `'group_mean'` | `group_mean`: each score less the group's mean. `leave_one_out`: less the mean of the others' (RLOO). `none`: the score itself. |
| `scale` | `str` | `'none'` | `none`, or `group_std`: divided by the standard deviation of the group's scores (GRPO). |
| `filter` | `str` | `'equal_scores'` | `equal_scores`: a group whose scores are all equal is skipped (DAPO's dynamic sampling). `none`: kept. |
| `tiebreak` | `float` | `0.0` | What the shortest episodes (`Episode.duration`) of a group whose every episode saturated its task score more, before the baseline. 0, every preset's: nothing, so how long an episode took never changes its score. |

### `Clip`

*class* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
class Clip
```

How the update's movement is bounded, through the ratio of each token's logprob now to the step's start.

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `str` | `'ratio'` | `none`; `ratio`: PPO's, the smaller of the ratio and the clipped ratio, each times the advantage; `weight`: the clipped ratio as a weight with no gradient, times the logprob (CISPO); `dual`: PPO's, and for a negative advantage no less than `dual` times it (dual-clip PPO). |
| `low` | `float` | `0.2` |  |
| `high` | `float` | `0.28` | The ratio is clipped to 1 - `low` .. 1 + `high` (asymmetric: DAPO's clip-higher). |
| `dual` | `float` | `3.0` | For `dual`: the bound for a negative advantage, in times it (above 1). |

### `Component`

*class* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
class Component
```

One component: its dotted key under `objective.`, the JSON types and strings it takes, the families that accept
it, and whether a running run takes it from its next step on.

| Field | Type | Default | Description |
|---|---|---|---|
| `key` | `str` | required |  |
| `types` | `tuple[str, ...]` | required |  |
| `families` | `frozenset[str]` | required |  |
| `changeable` | `bool` | required |  |
| `says` | `str` | required |  |
| `choices` | `tuple[str, ...]` | `()` |  |
| `least` | `float \| None` | `None` |  |
| `above` | `bool` | `False` | `least` itself is not taken. |

### `component`

*function* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
def component(key: str) -> Component | None
```

The component a dotted key names (`clip.low`, without `objective.`).

### `COMPONENTS`

*constant* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
COMPONENTS: tuple[Component, ...] = (Component('advantage.baseline', _S, _ADVANTAGED, False, 'What a score is measured against', ('group_mean', 'leave_one_out', 'none')), Component('advantage.scale', _S, _ADVANTAGED, False, 'What an advantage is divided by', ('none', 'group_std')), Component('advantage.filter', _S, _ADVANTAGED, False, 'Which groups are skipped', ('none', 'equal_scores')), Component('advantage.tiebreak', _F, _ADVANTAGED, False, 'What the shortest of a saturated group scores more; 0: nothing', least=0), Component('ratio', _S, _PG, False, "The ratio to the step's start", ('token', 'segment', 'none')), Component('clip.kind', _S, _PG, False, 'How the ratio is clipped', ('none', 'ratio', 'weight', 'dual')), Component('clip.low', _F, _PG, True, "The ratio's lower bound, below 1", least=0), Component('clip.high', _F, _PG, True, "The ratio's upper bound, above 1", least=0), Component('clip.dual', _F, _PG, True, "Dual clipping's bound, in times a negative advantage", least=1, above=True), Component('importance.correction', _S, _DISTILLED, False, 'The correction for where tokens were sampled', ('none', 'untruncated', 'truncate', 'mask')), Component('importance.level', _S, _DISTILLED, False, 'A weight per token or per segment', ('token', 'segment')), Component('importance.cap', _F, _DISTILLED, True, 'The largest weight', least=0, above=True), Component('importance.floor', _F, _DISTILLED, True, 'The smallest weight a mask keeps', least=0), Component('importance.paper_exact', _B, _DISTILLED, False, "No correction: the preset's loss as its paper writes it, for on-policy samples"), Component('kl.target', _S, _DISTILLED, False, 'What the KL penalty measures against', ('none', 'reference', 'old')), Component('kl.estimator', _S, _DISTILLED, False, 'How the KL is estimated', ('k1', 'k2', 'k3')), Component('kl.placement', _S, _DISTILLED, False, 'Where the KL penalty goes', ('loss', 'reward')), Component('kl.coefficient', _F, _DISTILLED, True, "The KL penalty's weight", least=0), Component('entropy.coefficient', _F, _PG, True, "The entropy bonus's weight"), Component('aggregate', _S, _AGGREGATED, False, 'How per-token losses become one', ('token_mean', 'segment_mean', 'segment_sum', 'constant')), Component('constant_tokens', _I, _AGGREGATED, False, 'The token count `constant` divides by', least=1), Component('reference', _S, frozenset({POLICY_GRADIENT, PREFERENCE, DISTILLATION}), False, 'The reference model', ('none', 'base')), Component('preference.loss', _S, _PREFERENCE, False, 'The preference loss', ('sigmoid', 'hinge', 'square', 'margin', 'odds_ratio', 'kto')), Component('preference.beta', _F, _PREFERENCE, True, "The preference loss's scale", least=0, above=True), Component('preference.margin', _F, _PREFERENCE, True, "SimPO's target margin"), Component('preference.length_normalized', _B, _PREFERENCE, False, "Each side's mean logprob, not its sum"), Component('preference.desirable', _F, _PREFERENCE, True, "KTO's weight of desirable examples", least=0), Component('preference.undesirable', _F, _PREFERENCE, True, "KTO's weight of undesirable examples", least=0), Component('likelihood.coefficient', _F, _PREFERENCE, True, 'A likelihood term beside the preference loss', least=0), Component('distillation.divergence', _S, _DISTILLED, False, 'The divergence from the teacher', ('reverse_kl', 'forward_kl', 'jsd')), Component('distillation.form', _S, _DISTILLED, False, "From the sampled tokens, or over the teacher's top-k", ('policy_gradient', 'top_k')), Component('distillation.top_k', _I, _DISTILLED, False, "The teacher's most likely tokens at each position", least=0), Component('distillation.temperature', _F, _DISTILLED, True, "A renormalized top-k divergence's temperature", least=0, above=True), Component('distillation.advantage_clip', _F, _DISTILLED, True, "The advantage's bound either side; 0: none", least=0), Component('distillation.beta', _F, _DISTILLED, True, "The teacher's weight in the JSD's mixture", least=0, above=True), Component('distillation.teachers', _T, _DISTILLED, False, 'The teacher channel of each route'), Component('distillation.coefficient', _F, _PG, True, 'A distillation term beside the policy gradient', least=0))
```

Every component, by dotted key under `objective.`.

### `composed`

*function* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
def composed(preset: str, overrides: Mapping[str, JsonValue] | None = None) -> tuple[Objective, list[tuple[str, str]]]
```

A preset with `overrides` in place, and what is wrong with it (`problems`), by dotted key. Raises `ValueError`
for a preset that does not exist, or an override that is no component or a value it does not take.

### `DEFAULT`

*constant* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
DEFAULT = PRESETS['default'].objective
```

### `Distillation`

*class* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
class Distillation
```

A divergence between a teacher's next-token distribution and the policy's at each sampled token
(`rollout_objectives.distillation`).

| Field | Type | Default | Description |
|---|---|---|---|
| `divergence` | `str` | `'reverse_kl'` | `reverse_kl`, KL(policy \|\| teacher); `forward_kl`, KL(teacher \|\| policy); `jsd`, GKD's generalized Jensen-Shannon divergence, `beta·KL(teacher \|\| m) + (1 - beta)·KL(policy \|\| m)` with `m = beta·teacher + (1 - beta)·policy`. |
| `form` | `str` | `'policy_gradient'` | `policy_gradient`: each sampled token's advantage is the teacher's logprob less the policy's at the step's start (no gradient), clipped to `advantage_clip`, and its loss the advantage times the policy's logprob (the reverse KL's gradient, estimated from the sampled tokens alone). `top_k`: the divergence over the teacher's top-k tokens at each sampled position: for `reverse_kl`, of the probabilities as they are there (MOPD's top-k form); for `forward_kl` and `jsd`, of both distributions renormalized over those tokens, at `temperature` (the rest of the vocabulary is dropped). |
| `top_k` | `int` | `0` | How many of the teacher's most likely tokens each sampled position carries (0: its logprob of the sampled token alone). |
| `temperature` | `float` | `1.0` | For a renormalized top-k divergence: both sides' logprobs are divided by it before renormalizing, and the divergence is multiplied by its square (Hinton et al., 2015). |
| `advantage_clip` | `float` | `0.0` | For the `policy_gradient` form: the advantage is clipped to -this .. this (0: not clipped). |
| `beta` | `float` | `0.5` | For `jsd`: the teacher's weight in the mixture, between 0 and 1. |
| `teachers` | `Mapping[str, str]` | `field(default_factory=dict[str, str], hash=False)` | The teacher channel that scores each episode, by route: an environment (`module:name`), one of its rows (`module:name/ROW`), or `*` for any other (`rollout_train.distillation.teacher_for`). One teacher scores each segment; teachers are never combined. |
| `coefficient` | `float` | `0.0` | For a policy gradient: a distillation term beside it, times this (0: none). |

### `Entropy`

*class* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
class Entropy
```

| Field | Type | Default | Description |
|---|---|---|---|
| `coefficient` | `float` | `0.0` | Each sampled position's entropy, times this, is taken from its loss (a bonus for keeping the policy spread). |

### `FAMILIES`

*constant* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
FAMILIES = (POLICY_GRADIENT, PREFERENCE, LIKELIHOOD, DISTILLATION)
```

Every family, the primary selector of an objective.

### `from_trainer_settings`

*function* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
def from_trainer_settings(said: Mapping[str, Any]) -> tuple[str | None, dict[str, JsonValue]]
```

The preset and overrides that a trainer's settings named the objective by (`LEGACY`), where they name it:
`objective = "policy_gradient"` is the `default` preset and `"likelihood"` is `sft`; `ratio = "segment"` is a
segment ratio and weight clipped to `segment_clip_low` and `segment_clip_high` (3e-4, 4e-4 by default), a mean over
segments; `clip_low` and `clip_high` bound a token ratio; `truncate` is the importance weight's cap (none: the
weight untruncated). A name other than those two is a preset's.

### `Importance`

*class* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
class Importance
```

The correction for where each token was sampled: the weight of its logprob at the step's start against the one
the engine recorded (`old / behavior`), a constant with no gradient. A turn may begin up to `max_lag` checkpoints
behind the newest, and the engine computes slightly differently from the trainer, so every policy-gradient preset
and the policy-gradient form of distillation truncate it at 2 by default.

| Field | Type | Default | Description |
|---|---|---|---|
| `correction` | `str` | `'truncate'` | `none`; `untruncated`: the weight (importance sampling); `truncate`: the weight, at most `cap` (truncated importance sampling); `mask`: the weight, and a token whose weight is outside `floor` .. `cap` is dropped (masked importance sampling). |
| `level` | `str` | `'token'` | `token`: a weight for each token. `segment`: one for the segment, the geometric mean of its tokens'. |
| `cap` | `float` | `2.0` |  |
| `floor` | `float` | `0.0` | For `mask`: the lowest weight kept. |
| `paper_exact` | `bool` | `False` | True: no correction (`correction = none` follows), so a preset's loss is its paper's exactly, which assumes on-policy samples: for comparing with a paper on a run whose samples are on-policy (`max_lag = 0`). |

### `Kl`

*class* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
class Kl
```

A penalty for the policy's divergence from a target, estimated on the sampled tokens.

| Field | Type | Default | Description |
|---|---|---|---|
| `target` | `str` | `'none'` | `none`; `reference`: the reference model (`Objective.reference`); `old`: the policy at the step's start. |
| `estimator` | `str` | `'k3'` | With `log_r` the target's logprob less the policy's: `k1` is `-log_r`, `k2` is `log_r² / 2`, `k3` is `exp(log_r) - 1 - log_r` (Schulman's estimators). `k1` is taken in the reward only: in the loss its gradient is the policy's logprob's, whose mean over the policy's own samples is 0 (`problems` refuses it). |
| `placement` | `str` | `'loss'` | `loss`: added to each token's loss, with its gradient. `reward`: taken from each token's advantage, with none. |
| `coefficient` | `float` | `0.0` |  |

### `LEGACY`

*constant* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
LEGACY = ('objective', 'ratio', 'clip_low', 'clip_high', 'segment_clip_low', 'segment_clip_high', 'truncate')
```

The trainer settings that once said the objective, which still say it (`from_trainer_settings`).

### `Likelihood`

*class* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
class Likelihood
```

| Field | Type | Default | Description |
|---|---|---|---|
| `coefficient` | `float` | `0.0` | A likelihood term beside a preference loss: the chosen side's mean negative logprob, times this (ORPO). |

### `Objective`

*class* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
class Objective
```

An objective, every component of it (those its family does not accept keep their defaults and mean nothing).

| Field | Type | Default | Description |
|---|---|---|---|
| `family` | `str` | `POLICY_GRADIENT` |  |
| `advantage` | `Advantage` | `field(default_factory=Advantage)` |  |
| `ratio` | `str` | `'token'` | `token`: each token's logprob now against the step's start. `segment`: one for the segment, the geometric mean of its tokens' (GSPO), its gradient spread over them. `none`: no ratio, the logprob itself (REINFORCE). |
| `clip` | `Clip` | `field(default_factory=Clip)` |  |
| `importance` | `Importance` | `field(default_factory=Importance)` |  |
| `kl` | `Kl` | `field(default_factory=Kl)` |  |
| `entropy` | `Entropy` | `field(default_factory=Entropy)` |  |
| `aggregate` | `str` | `'token_mean'` | How a minibatch's per-token losses become one: `token_mean`, over its sampled tokens; `segment_mean`, over each segment's tokens, then its segments; `segment_sum`, summed over each segment's tokens, then a mean over its segments (a sequence's logprob, as REINFORCE and RLOO take it); `constant`, summed over each segment's tokens and divided by `constant_tokens`, then a mean over its segments (Dr. GRPO). A preference loss is a mean over pairs or examples. |
| `constant_tokens` | `int` | `1024` | For `constant`: the fixed token count (the turn's token budget). |
| `reference` | `str` | `'none'` | The model the KL to the reference and a preference loss compare with: `base`, the model trained over (an adapter switched off; a frozen copy for a full-weight trainer); `none`. |
| `preference` | `Preference` | `field(default_factory=Preference)` |  |
| `likelihood` | `Likelihood` | `field(default_factory=Likelihood)` |  |
| `distillation` | `Distillation` | `field(default_factory=Distillation)` |  |
| `preset` | `str` | `'default'` | The preset it was resolved from (what it says, not what it is: the components are). |

**Methods**

- `def components(self) -> dict[str, JsonValue]` — The components its family accepts, by dotted key, with their values.
- `def get(self, key: str) -> JsonValue` — A component's value, by dotted key (`clip.low`).
- `def to_json(self) -> dict[str, JsonValue]` — What a run's start records: the preset, the family and the family's components.
- `@classmethod def from_json(cls, said: Mapping[str, JsonValue]) -> 'Objective'` — An objective as `to_json` recorded it.
- `def changed(self, changes: Mapping[str, JsonValue]) -> 'Objective'` — With `changes` (dotted keys, each a component that may change between steps), checked as any objective is
  (else `ValueError`). A change to the value a component has already changes nothing, and is not checked as an
  override (a step may be given every changeable component, at its value).
- `@property def needs_reference(self) -> bool` — Whether its loss reads the reference's logprobs.
- `@property def takes_importance(self) -> bool` — Whether an importance correction weighs its loss: a policy gradient's, or the policy-gradient form of
  distillation's, each a term of the sampled token (the top-k form's is a divergence over the teacher's top
  tokens at the position, which the sampled token's weight does not correct).
- `@property def needs_behaviour(self) -> bool` — Whether its loss reads the logprobs the engine recorded (an importance correction).
- `@property def distills(self) -> bool` — Whether its loss reads a teacher's logprobs: a distillation, or a policy gradient with a distillation term
  (its batch items are `rollout_train.trainer.Distilled`).
- `@property def needs_top(self) -> int` — How many of the teacher's most likely tokens each sampled position must carry (0: none).
- `@property def needs_distribution(self) -> bool` — Whether its loss reads the policy's logprobs of tokens other than the sampled ones: those of the teacher's
  top-k (the `top_k` form of distillation).
- `@property def needs_entropy(self) -> bool`
- `@property def labelled(self) -> bool` — Whether its batch items are labelled examples (KTO), rather than pairs.

### `objective_of`

*function* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
def objective_of(given: Any = None, legacy: Mapping[str, Any] | None = None) -> Objective
```

The objective a trainer is given: an `Objective`; a preset's name (or `policy_gradient`, `likelihood`); a table
of `preset` and overrides (dotted keys, or tables of them), or one recorded by `Objective.to_json` (it has
`family`); none for `default`. `legacy` are the trainer's other settings that once named the objective (`LEGACY`),
taken as overrides before the table's. Raises `ValueError` for one that is wrong.

### `Preference`

*class* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
class Preference
```

The preference loss, of each side's log-likelihood ratio to the reference (`rho`: the sum over its sampled
tokens of the policy's logprob less the reference's, or their mean with `length_normalized`).

| Field | Type | Default | Description |
|---|---|---|---|
| `loss` | `str` | `'sigmoid'` | For a pair, of `h = rho_chosen - rho_rejected` (`sigmoid(x)` the logistic function): `sigmoid`, `-log sigmoid(beta·h)` (DPO); `hinge`, `max(0, 1 - beta·h)`; `square`, `(h - 1/(2·beta))²` (IPO, beta as its tau); `margin`, `-log sigmoid(beta·h - margin)` of the likelihoods alone (SimPO); `odds_ratio`, `-beta·log sigmoid(log odds_chosen - log odds_rejected)` of the length-normalized likelihoods, beta weighing the term (ORPO). For a labelled example: `kto`, `desirable·(1 - sigmoid(beta·(rho - z)))` or `undesirable·(1 - sigmoid(beta·(z - rho)))`, `z` the mean of the minibatch's `rho` (no less than 0, no gradient). |
| `beta` | `float` | `0.1` |  |
| `margin` | `float` | `0.0` |  |
| `length_normalized` | `bool` | `False` |  |
| `desirable` | `float` | `1.0` |  |
| `undesirable` | `float` | `1.0` | KTO's weights of desirable and undesirable examples. |

### `Preset` {#rollout_trainobjectivespreset}

*class* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
class Preset
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `objective` | `Objective` | required |  |
| `source` | `str` | required | The paper it comes from. |
| `says` | `str` | required |  |

### `PRESETS`

*constant* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
PRESETS: Mapping[str, Preset] = {each.name: each for each in (_preset('default', 'this platform: Liu et al., 2025 (Dr. GRPO) for the advantage, Yu et al., 2025 (DAPO) for clip-higher and the token mean, Yao et al., 2025 for truncated importance sampling', "Dr. GRPO's advantages, DAPO's clip-higher and token mean, truncated importance sampling at 2"), _preset('reinforce', 'Williams, 1992', "the score times each segment's logprob: no baseline, ratio or clipping", advantage=Advantage(baseline='none', filter='none'), ratio='none', clip=_NO_CLIP, aggregate='segment_sum'), _preset('rloo', 'Ahmadian et al., 2024 (Back to Basics)', 'REINFORCE with a leave-one-out baseline', advantage=Advantage(baseline='leave_one_out', filter='none'), ratio='none', clip=_NO_CLIP, aggregate='segment_sum'), _preset('ppo_clip', 'Schulman et al., 2017', 'the token ratio clipped at 0.2 either side; advantages normalized within the group (no critic)', advantage=_GROUP_NORMALIZED, clip=Clip(low=0.2, high=0.2)), _preset('grpo', 'Shao et al., 2024 (DeepSeekMath)', "group mean and standard deviation; the token ratio clipped at 0.2; KL to the reference by k3 in the loss, at 0.04; a mean over each segment's tokens, then segments", advantage=_GROUP_NORMALIZED, clip=Clip(low=0.2, high=0.2), kl=Kl(target='reference', estimator='k3', placement='loss', coefficient=0.04), aggregate='segment_mean', reference='base'), _preset('dr_grpo', 'Liu et al., 2025 (Understanding R1-Zero-Like Training)', 'the group mean without the standard deviation; summed over tokens and divided by a constant; no KL', advantage=Advantage(filter='none'), clip=Clip(low=0.2, high=0.2), aggregate='constant', constant_tokens=3000), _preset('dapo', 'Yu et al., 2025 (DAPO)', 'clip-higher (0.2, 0.28); the token mean; groups of equal scores skipped; no KL', advantage=Advantage(scale='group_std', filter='equal_scores'), clip=Clip(low=0.2, high=0.28)), _preset('gspo', 'Zheng et al., 2025 (GSPO)', "the segment ratio, the geometric mean of its tokens', clipped to (3e-4, 4e-4); a mean over segments", advantage=_GROUP_NORMALIZED, ratio='segment', clip=Clip(low=0.0003, high=0.0004), importance=Importance(level='segment'), aggregate='segment_mean'), _preset('cispo', 'MiniMax, 2025 (MiniMax-M1)', 'the importance weight clipped above, with its gradient stopped, times the logprob: no update clipping', advantage=_GROUP_NORMALIZED, clip=Clip(kind='weight', low=1.0, high=3.0)), _preset('sft', 'supervised fine-tuning', "the sampled tokens' log-likelihood, each segment weighted by its advantage (1 for a dataset's)", family=LIKELIHOOD), _preset('dpo', 'Rafailov et al., 2023', 'the sigmoid loss over pairs, against the reference, at beta 0.1', family=PREFERENCE, reference='base', preference=Preference(loss='sigmoid', beta=0.1)), _preset('ipo', 'Azar et al., 2023', 'the square loss over pairs, against the reference', family=PREFERENCE, reference='base', preference=Preference(loss='square', beta=0.1, length_normalized=True)), _preset('simpo', 'Meng et al., 2024', 'the length-normalized margin loss, with no reference', family=PREFERENCE, preference=Preference(loss='margin', beta=2.0, margin=1.0, length_normalized=True)), _preset('kto', 'Ethayarajh et al., 2024', 'desirable and undesirable examples, unpaired, against the reference', family=PREFERENCE, reference='base', preference=Preference(loss='kto', beta=0.1)), _preset('orpo', 'Hong et al., 2024', "an odds-ratio term at 0.1 beside the chosen side's likelihood, with no reference", family=PREFERENCE, preference=Preference(loss='odds_ratio', beta=0.1, length_normalized=True), likelihood=Likelihood(coefficient=1.0)), _preset('on_policy_distillation', 'Agarwal et al., 2024 (GKD); Thinking Machines, 2025 (On-Policy Distillation)', "the reverse KL on the student's own samples, from the teacher's logprob of each sampled token: its advantage the teacher's logprob less the student's", family=DISTILLATION, distillation=Distillation(divergence='reverse_kl', form='policy_gradient')), _preset('distillation', 'Hinton et al., 2015; Kim and Rush, 2016', "the forward KL to the teacher's top-20 logprobs, renormalized over them, on the teacher's samples", family=DISTILLATION, importance=_NO_IMPORTANCE, distillation=Distillation(divergence='forward_kl', form='top_k', top_k=20)), _preset('mopd', 'Ma et al., 2026, MOPD: Multi-Teacher On-Policy Distillation (MiMo)', "each sampled token's advantage the teacher's logprob less the student's, clipped at 5; a mean over each segment's tokens; one teacher for each domain", family=DISTILLATION, distillation=Distillation(divergence='reverse_kl', form='policy_gradient', advantage_clip=5.0), aggregate='segment_mean'), _preset('mopd_top_k', 'Ma et al., 2026, MOPD: Multi-Teacher On-Policy Distillation (MiMo)', "MOPD's top-k form: the reverse KL over the teacher's top-64 tokens; a mean over each segment's tokens", family=DISTILLATION, importance=_NO_IMPORTANCE, distillation=Distillation(divergence='reverse_kl', form='top_k', top_k=64), aggregate='segment_mean'))}
```

Every preset, by name.

### `problems` {#rollout_trainobjectivesproblems}

*function* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
def problems(objective: Objective, overrides: Mapping[str, JsonValue] | None = None) -> list[tuple[str, str]]
```

What is wrong with an objective, as (dotted key, reason): an override of a component its family does not accept,
and combinations that mean nothing.

### `resolved`

*function* · `libraries/rollout-train/src/rollout_train/objectives.py`

```python
def resolved(preset: str, overrides: Mapping[str, JsonValue] | None = None) -> Objective
```

A preset with `overrides` (dotted keys under `objective.`) in place. Raises `ValueError` for a preset that does
not exist, an override that is no component or not of the family, a value it does not take, or a combination that
means nothing (`problems`).

## `rollout_train.run_settings`

A run's settings: the schema, layers, flags and files, a full copy, diffs.

### `Change`

*class* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
class Change
```

One key that differs between two settings: `before` or `after` is absent where the key was not given.

| Field | Type | Default | Description |
|---|---|---|---|
| `key` | `str` | required |  |
| `before` | `JsonValue` | `None` |  |
| `after` | `JsonValue` | `None` |  |
| `added` | `bool` | `False` |  |
| `removed` | `bool` | `False` |  |

### `diff`

*function* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
def diff(before: Mapping[str, JsonValue], after: Mapping[str, JsonValue]) -> list[Change]
```

What changed from `before` to `after`, key by key, in key order.

### `flattened`

*function* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
def flattened(table: Mapping[str, Any], prefix: str = '') -> dict[str, JsonValue]
```

Nested tables as dotted keys, down to a key the schema takes a table for (`channels.NAME.weights`).

### `from_file`

*function* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
def from_file(path: Path) -> dict[str, JsonValue]
```

Settings from a file: JSON (`.json`, where `null` unsets a key) or TOML, of dotted keys (`"trainer.rank" =
16`) or tables (`[trainer] rank = 16`), or both.

### `from_flags`

*function* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
def from_flags(given: Sequence[str]) -> dict[str, JsonValue]
```

`--set KEY=VALUE` flags, in order (a later one wins). A value is read as JSON (`null`, `3e-5`, `true`,
`["a", "b"]`, `"text"`), else as TOML (`{ a = 1 }`), else as the text it is (`rollout_qwen:qwen35`).

### `is_trainers`

*function* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
def is_trainers(key: str) -> bool
```

Whether `key` is one of the trainer's own settings (`trainer.rank`).

### `Key`

*class* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
class Key
```

One key of the schema.

| Field | Type | Default | Description |
|---|---|---|---|
| `pattern` | `str` | required | Dotted, `*` for a part that names something (`channels.*.model`). |
| `types` | `tuple[str, ...]` | required | The JSON types it takes: `int`, `float`, `bool`, `str`, `null`, `list`, `table`. |
| `default` | `JsonValue` | required |  |
| `changeable` | `bool` | required |  |
| `kinds` | `frozenset[str]` | required |  |
| `says` | `str` | required |  |
| `least` | `float \| None` | `None` | The smallest number it takes. |
| `choices` | `tuple[str, ...]` | `()` | The strings it takes, where it takes only some. |
| `above` | `bool` | `False` | `least` itself is not taken. |

**Methods**

- `def problem(self, value: JsonValue) -> str | None` — What is wrong with `value` for this key, in words; none when nothing is.

### `key_of`

*function* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
def key_of(key: str) -> Key | None
```

The schema's key that `key` is (`channels.policy.model` is `channels.*.model`); none for a key that is not in
it (a trainer's own setting among them).

### `KEYS`

*constant* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
KEYS: tuple[Key, ...] = (Key('kind', _S, 'train', False, EVERY, 'The kind of run', choices=KINDS), Key('name', _S + _N, None, False, EVERY, "What the run is called (the launch's name); never kept in a preset"), Key('environment', _S + _N, None, False, SAMPLING, 'The environment, `module:name`'), Key('groups', _I, 100, False, TRAINED | {'check'}, 'Groups it plays', least=1), Key('group_size', _I + _N, None, False, TRAINED, "Episodes of each group; none: its objective's", least=1), Key('seed', _I, 0, False, EVERY, 'The seed its draws start from', least=0), Key('start', _S + _N, None, False, EVERY, 'The checkpoint it trains from or evaluates; none: the base model'), Key('bookmark', _S + _N, None, False, TRAINING, 'A bookmark it moves to each checkpoint it makes'), Key('episodes_at_once', _I, 6, False, SAMPLING, 'Episodes it keeps work waiting for', least=1), Key('trainer.provider', _S + _N, None, False, TRAINING, 'The trainer, a `[trainers.NAME]` of the cluster'), Key('trainer.channel', _S, 'policy', False, TRAINED, 'The trained channel'), Key('trainer.model', _S + _N, None, False, TRAINING, "What the trainer trains over; none: the trained channel's"), Key('weights', _S + _N, None, False, TRAINING, 'What it trains: `lora` (an adapter) or `full` weights; none: what its trainer makes', choices=WEIGHTS), Key('channels.*.provider', _S + _N, None, False, SAMPLING, 'What samples the channel, an `[inference.NAME]`'), Key('channels.*.providers', ('list', 'null'), None, False, SAMPLING, 'Several providers serving it, in order'), Key('channels.*.routing', _S, 'spill', False, SAMPLING, 'How turns are shared among them', choices=ROUTING), Key('channels.*.weights', ('table', 'null'), None, False, SAMPLING, "Each provider's weight, for `weighted`"), Key('channels.*.model', _S + _N, None, False, RENDERING, "The model it serves, among its providers'"), Key('channels.*.renderer', _S + _N, None, False, RENDERING, 'The renderer, `module:name`'), Key('channels.*.thinking_tokens', _I + _N, None, False, SAMPLING, 'Thinking budget per turn', least=1), Key('channels.*.answer_tokens', _I + _N, None, False, SAMPLING, 'Room for the answer after it', least=1), Key('channels.*.replicas', _I + _N, None, False, SAMPLING, "Engine hosts; none: the provider's", least=1), Key('channels.*.bridge', _S, 'auto', False, SAMPLING, 'The bridge', choices=('auto', 'merge-quantize')), Key('channels.*.mode', _S + _N, None, False, SAMPLING, '`fixed` or `follows`; none: the trained channel serves what the run trains, another serves `fixed`', choices=('fixed', 'follows')), Key('channels.*.checkpoint', _S + _N, None, False, SAMPLING, 'What a `fixed` channel serves; none: the base model'), Key('channels.*.follows', _S + _N, None, False, SAMPLING, 'The channel a `follows` channel follows'), Key('channels.*.lag', _I, 0, False, SAMPLING, 'How many checkpoints behind it follows', least=0), Key('slots.*', _S, None, False, SAMPLING, "The channel a program's slot samples"), Key('self_judging', _B, False, False, SAMPLING, "Whether a judge may be bound to a channel serving the run's own"), Key('eval.suite', _S + _N, None, False, frozenset({'eval'}), 'The suite an eval plays, by name or `NAME@N`'), Key('eval.episodes', _I + _N, None, False, frozenset({'eval'}), 'Episodes of each start', least=1), Key('check.episodes', _I + _N, None, False, frozenset({'check'}), "Episodes of each group; none: a group's size", least=1), Key('imitation.dataset', _S + _N, None, False, frozenset({'imitate'}), 'The dataset, by name or id'), Key('imitation.limit', _I + _N, None, False, frozenset({'imitate'}), 'At most this many segments', least=1), Key('imitation.passes', _I, 1, False, frozenset({'imitate'}), 'Passes over the dataset', least=1), Key('imitation.warmup', _I, 0, False, frozenset({'imitate'}), 'Warm-up updates', least=0), Key('imitation.resume_optimizer', _B, False, False, frozenset({'imitate'}), "Go on from the start's optimizer"), Key('imitation.without', ('list',), [], False, frozenset({'imitate'}), 'Datasets whose segments are left out'), Key('groups_per_step', _I, 4, True, TRAINED, 'Groups a step waits for', least=1), Key('groups_ahead', _I + _N, None, True, TRAINED, 'At most this many groups in no step yet', least=1), Key('max_lag', _I, 1, True, TRAINED, 'Checkpoints behind the newest a turn may begin', least=0), Key('evals.suite', _S + _N, None, True, TRAINED, 'The suite its checkpoints play, by name or `NAME@N`'), Key('evals.every', _I, 1, True, TRAINED, 'Every this many steps', least=1), Key('evals.episodes', _I + _N, None, True, TRAINED, "Episodes of each start; none: the suite's", least=1), Key('limits.spend', ('float', 'null'), None, True, TRAINING | {'eval'}, "Dollars: the run ends once it spends this (hosted APIs' turns, its pods' hours); a training run whose step is estimated above it is refused", least=0), Key('limits.hours', ('float', 'null'), None, False, frozenset(KINDS), 'Hours: the run ends once it has run this long, its pods released', least=0), Key('objective.preset', _S, 'default', False, TRAINING, "The objective's preset", choices=tuple(PRESETS)), *(Key(f'objective.{each.key}', (*each.types, 'null'), None, each.changeable, TRAINING, f"{each.says}; none: the preset's", least=each.least, choices=each.choices, above=each.above) for each in COMPONENTS))
```

Every key a run takes, beside the trainer's own (`trainer.FIELD`).

### `KINDS`

*constant* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
KINDS = ('train', 'eval', 'imitate', 'check')
```

### `layered`

*function* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
def layered(*layers: Mapping[str, JsonValue] | None) -> RunSettings
```

Settings given in layers, each over the ones before (a preset's, a file's, the flags'): a key of a later layer
replaces the same key of an earlier one.

### `objective_in`

*function* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
def objective_in(settings: RunSettings) -> Objective
```

The objective a run's settings ask for: their preset with each component they give. Raises `ValueError` for one
that is wrong (`rollout_train.objectives.resolved`).

### `recorded`

*function* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
def recorded(settings: RunSettings, trainer: Sequence[SettingSpec] = (), preset: str | None = None) -> dict[str, JsonValue]
```

What a run's start records of its settings: a full copy, fixed and changeable (every key with its value,
defaults included); for a run that trains, the objective they resolve to (`objective_in`), so that what it trains
with never depends on what a preset means later; and, as provenance only, the preset version they came from
(`NAME@N`).

### `RunSettings`

*class* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
class RunSettings
```

A run's settings, as given (`values`), answering with the schema's defaults for what was not. Trainer settings
that name the objective are held as the `objective.*` keys they say (`current`).

| Field | Type | Default | Description |
|---|---|---|---|
| `values` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` |  |

**Methods**

- `def get(self, key: str, default: JsonValue = None) -> JsonValue` — A key's value: as given, else the schema's default, else `default` (a trainer's own setting).
- `@property def kind(self) -> str`
- `@property def trained(self) -> str | None` — The trained channel, for a training run.
- `@property def channels(self) -> list[str]` — Every channel the settings name, in the order first named.
- `def providers(self, channel: str) -> tuple[str, ...]` — The providers of a channel, in order (`providers`, or the one `provider`).
- `@property def trainer_model(self) -> str | None` — What the trainer trains over: `trainer.model`, else the trained channel's model.
- `def mode(self, channel: str) -> str` — `trained` for the trained channel, else its `mode` (`fixed` by default).
- `def split(self, trainer: Sequence[SettingSpec] = ()) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]` — The settings in full (every key of the schema for the run's kind and channels, with defaults, and the
  trainer's own with theirs), as fixed ones and changeable ones.

### `shortcuts`

*function* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
def shortcuts(*, model: str | None = None, provider: str | None = None, renderer: str | None = None, trainer: str | None = None, channel: str = 'policy') -> dict[str, JsonValue]
```

What `--model`, `--provider`, `--renderer` and `--trainer` set, for the channel `--channel` names.

### `WEIGHTS`

*constant* · `libraries/rollout-train/src/rollout_train/run_settings.py`

```python
WEIGHTS = ('lora', 'full')
```

What a run trains: a LoRA (low-rank adapter) over its model, or every weight.

## `rollout_train.stores`

The ledger and the blob store a cluster config names, opened on this node.

### `blobs_at`

*function* · `libraries/rollout-train/src/rollout_train/stores.py`

```python
def blobs_at(cluster: 'Cluster', store: str | None = None) -> dict[str, JsonValue]
```

Where a cluster's blob store is (its `[blobs]`, or the store `[stores.NAME]` names), as `opened` opens it: a
directory of files made absolute. Raises `KeyError` for a store the cluster does not name.

### `cluster_ledger`

*function* · `libraries/rollout-train/src/rollout_train/stores.py`

```python
def cluster_ledger(cluster: Mapping[str, Any]) -> 'Ledger'
```

The ledger a cluster config names (given as JSON, `Cluster.described`), opened on this node: what a ledger's
location `ledger_at` gives names (`rollout_train.ledger.opened`).

### `described`

*function* · `libraries/rollout-train/src/rollout_train/stores.py`

```python
def described(where: Mapping[str, Any], name: str | None = None) -> dict[str, JsonValue]
```

A blob store as a page says it, from its location (`blobs_at`, `location`): its `name` (`[stores.NAME]`; none:
`[blobs]`), its `kind` (`files`, or `module:name`), and its `bucket` and `prefix` or its `directory`; never a key,
nor the variables one is read from.

### `FILES`

*constant* · `libraries/rollout-train/src/rollout_train/stores.py`

```python
FILES = 'rollout.harness.blobs:FileBlobStore'
```

The store of files in a directory (`{"kind": FILES, "directory": …}`).

### `for_pods`

*function* · `libraries/rollout-train/src/rollout_train/stores.py`

```python
def for_pods(cluster: 'Cluster', store: str | None, *, writes: bool, environ: Mapping[str, str] | None = None) -> tuple[dict[str, JsonValue], dict[str, str]]
```

Where a pod finds a store, and the variables it is given with the store's key: the store's own key for a pod
that writes (a trainer's), its read-only key (`reader`) for one that only reads, where it names one. The key is
read here (`environ`, by default this process's environment); the location names the variables the pod reads it
from. Raises `ValueError` where the key is named and not set here.

### `ledger_at`

*function* · `libraries/rollout-train/src/rollout_train/stores.py`

```python
def ledger_at(cluster: 'Cluster') -> dict[str, JsonValue]
```

Where a cluster's ledger is, as a ledger's location (`rollout_train.ledger.opened`) that names the cluster
config: whoever opens it reads the URL, and any secret it is behind, on its own node.

### `ledger_of`

*function* · `libraries/rollout-train/src/rollout_train/stores.py`

```python
def ledger_of(cluster: 'Cluster', environ: Mapping[str, str] | None = None) -> 'Ledger'
```

The ledger a cluster's config names, opened on this node (`ledger_url`, `opened_ledger`).

### `ledger_url`

*function* · `libraries/rollout-train/src/rollout_train/stores.py`

```python
def ledger_url(cluster: 'Cluster', environ: Mapping[str, str] | None = None) -> str
```

The URL of the ledger a cluster names: its `[ledger] url`, or the value of the secret it names, read now on
this node. Raises `ClusterError` where the secret is not set here, or the URL is not a database's (saying where
it came from, never the URL: it may hold a password).

### `location`

*function* · `libraries/rollout-train/src/rollout_train/stores.py`

```python
def location(store: Mapping[str, Any], directory: Path) -> dict[str, Any]
```

Where a blob store is (`store`: a `[blobs]` table, `kind` and the store's settings; empty: files under
`directory`), without any setting that looks like a credential (the names of the variables one is read from,
`…_env`, are kept).

### `opened`

*function* · `libraries/rollout-train/src/rollout_train/stores.py`

```python
def opened(where: Mapping[str, Any]) -> Blobs
```

The blob store a location names.

### `opened_ledger`

*function* · `libraries/rollout-train/src/rollout_train/stores.py`

```python
def opened_ledger(url: str, token: 'Secret | None' = None) -> 'Ledger'
```

The ledger at `url`: a database (`sqlite:///…`, `postgresql://…`), or the ledger service (`http(s)://…`) with
the token `token` names.

### `store_named`

*function* · `libraries/rollout-train/src/rollout_train/stores.py`

```python
def store_named(cluster: 'Cluster', where: Mapping[str, Any]) -> str | None
```

The name of the cluster's store a location is (`[stores.NAME]`), where it is one of them; none for `[blobs]`
or a store the cluster does not name.

### `Stores`

*class* · `libraries/rollout-train/src/rollout_train/stores.py`

```python
class Stores
```

The ledger and the blob store, opened, and where the blob store is (`location`: as any process opens it, for a
run's `starts` record).

| Field | Type | Default | Description |
|---|---|---|---|
| `ledger` | `'Ledger'` | required |  |
| `blobs` | `Blobs` | required |  |
| `location` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` |  |

**Methods**

- `@classmethod def open(cls, cluster: 'Cluster', environ: Mapping[str, str] | None = None, *, store: str | None = None) -> 'Stores'` — The stores a cluster's config names, opened on this node: the ledger from `[ledger]` (its URL read from the
  secret it names, where it names one), the blob store from `[blobs]` (or `[stores.NAME]`, with `store`).
  Raises `ClusterError` where the ledger's URL is not set here or is neither a database's nor the ledger
  service's.
- `def writing_to(self, cluster: 'Cluster', store: str | None) -> 'Stores'` — The same ledger, with blobs written to `store` (`[stores.NAME]`; none: the default).
- `@property def checkpoints(self) -> 'Checkpoints'`
- `@property def registry(self) -> 'Registry'` — Run names, bookmarks, and dataset and suite names, beside the ledger.
- `@property def presets(self) -> 'Presets'`

## `rollout_train.ledger_service`

The ledger over HTTP: the service, the client every role can use, pods' tokens.

### `app`

*function* · `libraries/rollout-train/src/rollout_train/ledger_service/service.py`

```python
def app(ledger: Ledger, secret: Callable[[], str | None], *, more: Mapping[str, tuple[Handler, Handler | None]] | None = None, honoured: Callable[[Ledger, Scope], Awaitable[bool]] | None = None) -> 'Starlette'
```

The service over `ledger`, its tokens checked against the platform's token, which `secret` reads (at each
request: a token rotated where it is kept is honoured at once). `more` adds operations. `honoured` says whether a
pod's token is honoured now (whether the pod's lease names its run).

### `Conflict` {#rollout_trainledger_serviceconflict}

*class* · `libraries/rollout-train/src/rollout_train/ledger_service/wire.py`

```python
class Conflict(Exception)
```

A compare-and-set that found the row changed since it was read.

### `Forbidden`

*class* · `libraries/rollout-train/src/rollout_train/ledger_service/scopes.py`

```python
class Forbidden(Exception)
```

The token may not do what was asked.

### `HttpLedger`

*class* · `libraries/rollout-train/src/rollout_train/ledger_service/client.py`

```python
class HttpLedger
```

The ledger at the service `url`, with the token read (at each request, never kept) from the environment
variable `token_env` or the file `token_file`, or given as `token`.

**Methods**

- `def __init__(self, url: str, *, token_env: str | None = None, token_file: str | None = None, token: str | None = None, client: httpx.AsyncClient | None = None, deadline: float = DEADLINE, attempt: float = ATTEMPT) -> None`
- `def use(self, token: str) -> None` — Send `token` from now on (a pod given a token for the run that took it).
- `def token(self) -> str | None`
- `async def call(self, operation: str, args: Mapping[str, Any] | None = None) -> tuple[Any, bool]` — Do `operation` (`STORE/OPERATION`) with `args`: its result, and whether an attempt before the one answered
  may have been done (its answer lost).
- `async def result(self, operation: str, args: Mapping[str, Any] | None = None) -> Any`
- `async def take(self, scope: str) -> Fence`
- `async def append(self, table: str, key: str, record: JsonValue, fence: Fence) -> bool`
- `async def append_returning(self, table: str, key: str, record: JsonValue, fence: Fence) -> Appended`
- `async def read(self, table: str) -> dict[str, JsonValue]`
- `async def tables(self) -> list[str]`
- `async def read_all(self, *, leaving_out: str | None = None) -> dict[str, dict[str, JsonValue]]`
- `async def fences(self) -> dict[str, int]`
- `async def now(self) -> float` — The service's clock, in seconds since the epoch.
- `@property def registry(self) -> 'HttpRegistry'`
- `@property def launches(self) -> 'HttpLaunches'`
- `@property def presence(self) -> 'HttpPresence'`
- `@property def desired_settings(self) -> 'HttpDesiredSettings'`
- `@property def sandboxes(self) -> 'HttpLeases'`
- `@property def presets(self) -> 'HttpPresets'`
- `@property def environment_versions(self) -> 'HttpEnvironmentVersions'`
- `@property def pods(self) -> 'HttpPodLeases'`
- `async def aclose(self) -> None`
- `def close(self) -> None`

### `LedgerUnreachable`

*class* · `libraries/rollout-train/src/rollout_train/ledger_service/client.py`

```python
class LedgerUnreachable(Exception)
```

The ledger service did not answer within the deadline: what was asked may or may not have been done.

### `PLATFORM`

*constant* · `libraries/rollout-train/src/rollout_train/ledger_service/scopes.py`

```python
PLATFORM = Scope()
```

### `pod_token`

*function* · `libraries/rollout-train/src/rollout_train/ledger_service/scopes.py`

```python
def pod_token(secret: str, pod: str, run: str) -> str
```

A token for pod `pod` serving run `run`, signed with the platform's token `secret`.

### `Scope`

*class* · `libraries/rollout-train/src/rollout_train/ledger_service/scopes.py`

```python
class Scope
```

Whose a token is: the platform's (`pod` None), or a pod's, for the run it serves.

| Field | Type | Default | Description |
|---|---|---|---|
| `pod` | `str \| None` | `None` |  |
| `run` | `str \| None` | `None` |  |

**Methods**

- `@property def platform(self) -> bool`
- `@property def key(self) -> str` — Who it is, as the service keeps answers to retried requests by.

### `scope_of`

*function* · `libraries/rollout-train/src/rollout_train/ledger_service/scopes.py`

```python
def scope_of(token: str | None, secret: str) -> Scope | None
```

Whose `token` is, checked against the platform's token `secret`; None for a token that is neither the platform's
nor a pod's signed with it.

## `rollout_train.presets`

Named, versioned run settings beside the ledger.

### `DatabasePresets`

*class* · `libraries/rollout-train/src/rollout_train/presets.py`

```python
class DatabasePresets
```

`Presets` in the `presets` table of a database (a row per version, keyed by name and version: inserting a
version that is there fails, and the next number is tried).

**Methods**

- `def __init__(self, database: 'Database') -> None`
- `async def all(self) -> list[Preset]`
- `async def versions(self, name: str) -> list[Preset]`
- `async def get(self, reference: str) -> Preset | None`
- `async def save(self, name: str, settings: Mapping[str, JsonValue], note: str = '') -> Preset`
- `async def delete(self, name: str) -> Preset`

### `FilePresets`

*class* · `libraries/rollout-train/src/rollout_train/presets.py`

```python
class FilePresets
```

`Presets` in a directory: `NAME/N.json` for each version, each written beside its place and linked into it
only if no version N is there (the compare-and-set), so writers need no lock.

**Methods**

- `def __init__(self, directory: Path) -> None`
- `async def all(self) -> list[Preset]`
- `async def versions(self, name: str) -> list[Preset]`
- `async def get(self, reference: str) -> Preset | None`
- `async def save(self, name: str, settings: Mapping[str, JsonValue], note: str = '') -> Preset`
- `async def delete(self, name: str) -> Preset`

### `parsed` {#rollout_trainpresetsparsed}

*function* · `libraries/rollout-train/src/rollout_train/presets.py`

```python
def parsed(reference: str) -> tuple[str, int | None]
```

`NAME` or `NAME@N`, as the name and the version (none: the newest).

### `Preset` {#rollout_trainpresetspreset}

*class* · `libraries/rollout-train/src/rollout_train/presets.py`

```python
class Preset
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `version` | `int` | required | 1, 2, …: each save is a new version. |
| `settings` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` |  |
| `saved` | `float` | `0.0` |  |
| `note` | `str` | `''` |  |
| `deleted` | `bool` | `False` | A version that says the preset was deleted: its name points to nothing from here on. |

**Methods**

- `@property def id(self) -> str` — `NAME@N`: what a run records as where its settings came from.

### `Presets`

*class* · `libraries/rollout-train/src/rollout_train/presets.py`

```python
class Presets(Protocol)
```

**Methods**

- `async def all(self) -> list[Preset]` — Every preset's newest version, by name, leaving out the deleted ones.
- `async def versions(self, name: str) -> list[Preset]` — Every version of a preset saved, oldest first (deletions left out).
- `async def get(self, reference: str) -> Preset | None` — A preset by name (its newest version, none if it was deleted) or one version (`NAME@N`).
- `async def save(self, name: str, settings: Mapping[str, JsonValue], note: str = '') -> Preset` — Save settings as the preset's next version. Raises `ValueError` for a name that cannot be one, or a key a
  preset cannot hold.
- `async def delete(self, name: str) -> Preset` — Mark a preset deleted (its versions stay readable). Raises `KeyError` for a preset there is none of.

### `presets_of`

*function* · `libraries/rollout-train/src/rollout_train/presets.py`

```python
def presets_of(ledger: object) -> Presets | None
```

The presets beside a ledger: a table in a database ledger's database, a directory beside a ledger of files
(`presets`, in its directory), the service's for a ledger reached through it (`HttpLedger.presets`); none beside
any other.

## `rollout_train.published`

Versions of environments imported from their source, beside the ledger.

### `DatabaseEnvironmentVersions`

*class* · `libraries/rollout-train/src/rollout_train/published.py`

```python
class DatabaseEnvironmentVersions
```

`EnvironmentVersions` in the `environment_versions` table of a database: a row per version, keyed by its id
(inserting one that is there fails, and the one there is read).

| Field | Type | Default | Description |
|---|---|---|---|
| `COLUMNS` |  | `('version', 'name', 'source', 'ref', 'commit_id', 'subdirectory', 'entry_point', 'blob', 'runtime_env', 'dependencies', 'description', 'checked', 'imported')` |  |

**Methods**

- `def __init__(self, database: 'Database') -> None`
- `async def all(self) -> list[EnvironmentVersion]`
- `async def get(self, reference: str) -> EnvironmentVersion | None`
- `async def record(self, version: EnvironmentVersion) -> EnvironmentVersion`

### `environment_versions_of`

*function* · `libraries/rollout-train/src/rollout_train/published.py`

```python
def environment_versions_of(ledger: object) -> EnvironmentVersions | None
```

The published versions beside a ledger: a table in a database ledger's database, a directory beside a ledger
of files (`environment_versions`, in its directory), the service's for a ledger reached through it
(`HttpLedger.environment_versions`); none beside any other.

### `EnvironmentVersion`

*class* · `libraries/rollout-train/src/rollout_train/published.py`

```python
class EnvironmentVersion
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | The environment's name: its entry point's name in the project's `rollout.environments` group, else the project's name. |
| `version` | `str` | required | The SHA-256 of its source's zip. |
| `source` | `str` | required | The git URL it was fetched from. |
| `ref` | `str \| None` | required | The branch, tag or commit asked for; none: the default branch. |
| `commit` | `str` | required | The commit the ref was when it was fetched. |
| `subdirectory` | `str` | required | The project's directory in the repository (empty: its root). |
| `entry_point` | `str` | required | `module:name`: what makes the environment, imported in its runtime environment. |
| `blob` | `Mapping[str, JsonValue]` | required | The zip's blob reference (`rollout.contracts.BlobReference`, as JSON). |
| `runtime_env` | `Mapping[str, JsonValue]` | required | The Ray runtime environment its code runs in: `working_dir` (the zip, where Ray fetches it), `env_vars` (the project's `src` on the path, for a project laid out so), and `uv` (the dependencies the platform does not hold). |
| `dependencies` | `Sequence[str]` | `()` | The project's dependencies, as its `pyproject.toml` declares them. |
| `description` | `Mapping[str, JsonValue]` | `field(default_factory=dict[str, JsonValue])` | What the environment says of itself: its version, description, rows, eval data and curriculum. |
| `check` | `Sequence[Mapping[str, JsonValue]]` | `()` | What its check found: each finding's check, whether it passed, and what it said. |
| `imported` | `float` | `0.0` |  |

**Methods**

- `@property def reference(self) -> str` — `NAME@VERSION`: how launches, runs and suites name it.
- `def to_json(self) -> dict[str, Any]`

### `EnvironmentVersions`

*class* · `libraries/rollout-train/src/rollout_train/published.py`

```python
class EnvironmentVersions(Protocol)
```

**Methods**

- `async def all(self) -> list[EnvironmentVersion]` — Every version, the newest imported first.
- `async def get(self, reference: str) -> EnvironmentVersion | None` — A version by its id, or by `NAME@VERSION` (none where the name is not its).
- `async def record(self, version: EnvironmentVersion) -> EnvironmentVersion` — Record a version, unless one of its id is there: the version as it is kept (the one there, if there was
  one).

### `FileEnvironmentVersions`

*class* · `libraries/rollout-train/src/rollout_train/published.py`

```python
class FileEnvironmentVersions
```

`EnvironmentVersions` in a directory: `VERSION.json` for each, written beside its place and linked into it only
if no file of its id is there, so writers need no lock.

**Methods**

- `def __init__(self, directory: Path) -> None`
- `async def all(self) -> list[EnvironmentVersion]`
- `async def get(self, reference: str) -> EnvironmentVersion | None`
- `async def record(self, version: EnvironmentVersion) -> EnvironmentVersion`

### `is_published`

*function* · `libraries/rollout-train/src/rollout_train/published.py`

```python
def is_published(environment: str) -> bool
```

Whether an environment's name is a published version's (`NAME@VERSION`), not a built-in's (`module:name`).

### `loaded` {#rollout_trainpublishedloaded}

*function* · `libraries/rollout-train/src/rollout_train/published.py`

```python
async def loaded(environment: str, ledger: object) -> tuple[Any, EnvironmentVersion | None]
```

An environment by its name, imported here: a built-in one by `module:name`; a published one (`NAME@VERSION`) by
its version's entry point, which imports where this process runs in the version's runtime environment (a Ray job
given its `runtime_env`), with the version. Raises `KeyError` for a published version the ledger does not keep,
and whatever importing raises.

### `parsed` {#rollout_trainpublishedparsed}

*function* · `libraries/rollout-train/src/rollout_train/published.py`

```python
def parsed(reference: str) -> tuple[str, str] | None
```

`NAME@VERSION` as its name and version; none for anything else.

### `provenance`

*function* · `libraries/rollout-train/src/rollout_train/published.py`

```python
def provenance(version: EnvironmentVersion) -> dict[str, JsonValue]
```

What a run's start records of the published version it plays: where its source came from and what it ran.

### `short`

*function* · `libraries/rollout-train/src/rollout_train/published.py`

```python
def short(version: str) -> str
```

A version's id, in a few characters.

## `rollout_train.publishing`

Importing an environment from git: fetched, stored, checked on Ray, recorded.

### `checked_on_ray`

*function* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
async def checked_on_ray(jobs: str, entry_point: str, runtime_env: Mapping[str, JsonValue], *, within: float = CHECKING, every: float = 1.0) -> dict[str, Any]
```

Check an environment in a Ray job in its runtime environment (`report`, run by `python -m
rollout_train.publishing check`), on the cluster whose job server is `jobs`: what it found (`findings`), what the
environment says of itself (`described`), and the job's Python (`python`, and where it imported `rollout` and
`rollout_train` from: `platform`). Raises `Refused` where the job's Python environment is not built, the entry
point does not load, a check fails, or the job does not end within `within` seconds.

### `checked_with`

*function* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
async def checked_with(package: str, data: bytes, scratch: Path) -> AsyncGenerator[str]
```

The zip a check's job is handed: `package` where it is a file here (a store of files'), else `data` written
to a file under `scratch` for as long as the check runs. Ray's job submitter uploads a local zip to the cluster, so
the cluster's nodes never fetch it from the store.

### `entry_point_of`

*function* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
def entry_point_of(project: Mapping[str, Any], given: str | None) -> tuple[str, str]
```

The environment's name and entry point (`module:name`) of a project (its `[project]` table): `given` (an entry
point, or its name in `GROUP`), else the one environment the project declares in `GROUP`. The name is the entry
point's in `GROUP`, else the project's. Raises `Refused` where there is none, or several and none given.

### `EXCLUDED`

*constant* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
EXCLUDED = frozenset({'.git', '.venv', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.ipynb_checkpoints'})
```

Directories never packed.

### `fetched`

*function* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
async def fetched(url: str, ref: str | None, into: Path) -> tuple[Path, str]
```

A shallow clone of `url` at `ref` (a branch or tag; a commit, where the server gives one by its id; none: the
default branch) in `into`, and the commit it is. Raises `Refused` saying why it does not clone.

### `GROUP`

*constant* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
GROUP = 'rollout.environments'
```

The entry-point group a project declares its environments in.

### `Importer`

*class* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
class Importer
```

Where imports are made: the blob store their zips go to, the Ray cluster's job server their checks run on, and
the directory clones are read in.

| Field | Type | Default | Description |
|---|---|---|---|
| `blobs` | `Blobs` | required |  |
| `jobs` | `str` | required |  |
| `scratch` | `Path` | required |  |

**Methods**

- `@classmethod def of(cls, cluster: 'Cluster') -> 'Importer'` — A cluster's: its `[blobs]`, its `[ray] jobs`, and `imports` under its `[scratch]`.

### `MARK`

*constant* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
MARK = 'rollout-check: '
```

What the check's job begins the line it says what it found with.

### `missing`

*function* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
def missing(dependencies: Sequence[str]) -> list[str]
```

The requirements of `dependencies` this Python does not satisfy: each whose distribution is not installed, is
installed at a version its specifier leaves out, is asked for at a URL, or lacks what an extra asked for needs.
Requirements whose markers do not hold here are left out. Raises `Refused` for one that is no requirement.

### `packed` {#rollout_trainpublishingpacked}

*function* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
def packed(directory: Path) -> bytes
```

A project's files as a zip: every regular file under `directory` but those in `EXCLUDED` directories and
compiled bytecode, by its path relative to `directory`, stored uncompressed in sorted order with fixed times and
modes (executable or not). Raises `Refused` for a zip larger than `LARGEST`.

### `Project`

*class* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
class Project
```

A project as an import reads it: its directory, its environment's name and entry point, where its module is
imported from (`""` or `"src"`), and its dependencies.

| Field | Type | Default | Description |
|---|---|---|---|
| `directory` | `Path` | required |  |
| `name` | `str` | required |  |
| `entry_point` | `str` | required |  |
| `root` | `str` | required |  |
| `dependencies` | `tuple[str, ...]` | required |  |

### `project_of`

*function* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
def project_of(clone: Path, subdirectory: str, given: str | None) -> Project
```

The project in a clone's `subdirectory` (empty: its root), with its environment's entry point (`given`, else
the one it declares: `entry_point_of`). Raises `Refused` for a directory that is not in the clone or holds no
`pyproject.toml`, a project with no entry point, or an entry point whose module is not in it.

### `publish` {#rollout_trainpublishingpublish}

*function* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
async def publish(source: Source, *, versions: EnvironmentVersions, blobs: Blobs, jobs: str, scratch: Path, said: Said = _quiet, within: float = CHECKING) -> Published
```

Import an environment from git: fetch it, read it, pack it, store it, check it in a Ray job on the cluster
whose job server is `jobs`, and record it (the module's docstring). `scratch` holds the clone while it is read.
`said` is told each stage as it begins. Raises `Refused` saying why an import cannot be made.

### `Published`

*class* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
class Published
```

What an import made: the version, and whether it was recorded already (the same source imported before).

| Field | Type | Default | Description |
|---|---|---|---|
| `version` | `EnvironmentVersion` | required |  |
| `existing` | `bool` | required |  |

### `Refused` {#rollout_trainpublishingrefused}

*class* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
class Refused(ValueError)
```

An import that cannot be made, and why.

### `report`

*function* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
def report(entry_point: str) -> dict[str, Any]
```

What the check's job says of an environment, imported here by its entry point: whether it loaded (and why not),
the findings of `rollout_train.check.checked` and of an episode answered by a scripted model (where its program
imports no tool set and declares no sandbox, which an import cannot serve), what it says of itself, and this
Python and where it imported `rollout` and `rollout_train` from.

### `runtime_env_of` {#rollout_trainpublishingruntime_env_of}

*function* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
def runtime_env_of(package: str, root: str, dependencies: Sequence[str]) -> dict[str, JsonValue]
```

The Ray runtime environment a version's code runs in: `package` (the zip, where Ray fetches it) as its
`working_dir`; `root` (`src`, for a project that keeps its packages there) on `PYTHONPATH`, relative to that
directory, which every process of the job starts in; and `dependencies` (those the platform does not hold) for uv
to install over a copy of the platform's Python. Ray names that copy by the dependencies, so versions that need the
same ones share it.

### `Source`

*class* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
class Source
```

What to import: a git URL; the branch, tag or commit (none: the default branch); the project's directory in
the repository; its entry point (`module:name`, or its name in `GROUP`; none: the one it declares).

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | required |  |
| `ref` | `str \| None` | `None` |  |
| `subdirectory` | `str` | `''` |  |
| `entry_point` | `str \| None` | `None` |  |

### `stored` {#rollout_trainpublishingstored}

*function* · `libraries/rollout-train/src/rollout_train/publishing.py`

```python
async def stored(blobs: Blobs, data: bytes) -> tuple[BlobReference, str]
```

Store a project's zip, and say where Ray fetches it as a runtime environment's `working_dir`: for a store that
names copies of its blobs (`with_extension`: in S3, `s3://BUCKET/KEY.zip`), that; for a store of files, a `.zip`
beside its blobs (`packages/VERSION.zip`), which whoever submits the job uploads. Raises `Refused` for a store Ray
cannot be handed a blob of.

## `rollout_train.validation`

One pure check of a run's settings against a cluster, with its rule table.

### `check`

*function* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
def check(settings: RunSettings, cluster: Cluster, environment: EnvironmentFacts | None = None, ledger: LedgerFacts | None = None, *, model: ModelFacts | None = None) -> list[Finding]
```

Everything wrong with a run's settings on this cluster, given what is known of its environment, the ledger and
the trained model's size; empty when nothing is. A finding whose `refuses` is false is a note: the run may go.

### `CheckpointFacts`

*class* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
class CheckpointFacts
```

A checkpoint a run's settings name (its start, a fixed channel's), as the ledger has it.

| Field | Type | Default | Description |
|---|---|---|---|
| `reference` | `str` | required |  |
| `exists` | `bool` | `True` |  |
| `released` | `bool` | `False` | Its weights were deleted. |
| `formats` | `frozenset[str]` | `frozenset()` | The formats its files are in (`rollout_train.bridges.format_of`). |
| `model` | `str \| None` | `None` | The model it was trained over. |

### `completed`

*function* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
def completed(settings: RunSettings, cluster: Cluster) -> RunSettings
```

A run's settings with what follows from them said: what it trains (`with_weights`) and each channel's renderer
(`with_renderers`), as its start records them.

### `EnvironmentFacts`

*class* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
class EnvironmentFacts
```

What the environment's worker says of it, asked beforehand.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `loads` | `bool` | `True` |  |
| `why` | `str` | `''` | Why it does not load, where it does not. |
| `sandboxes` | `frozenset[str]` | `frozenset()` | The sandbox kinds its programs need. |
| `tool_sets` | `frozenset[str]` | `frozenset()` | The tool sets its programs import by name that are served elsewhere (`[tools.NAME]`). |
| `slots` | `frozenset[str] \| None` | `None` | Its programs' slots (none: not known). |
| `untrained` | `frozenset[str]` | `frozenset()` | Those of its slots that are not trained (a judge, a fixed opponent): each must be bound by name. |
| `judges` | `frozenset[str]` | `frozenset()` | Those of its slots that judge: bound to a channel serving the run's own checkpoints only with `self_judging`. |
| `episodes_per_group` | `int \| None` | `None` |  |
| `turns_per_episode` | `float \| None` | `None` |  |
| `samples_per_turn` | `float` | `1.0` | Samples a turn takes: one for each model slot that samples in it (each agent of a team). |
| `prompt_tokens` | `int \| None` | `None` | Prompt tokens of a sample, on average: with the three above, what a step's spend is estimated from. |

### `estimated_spend`

*function* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
def estimated_spend(settings: RunSettings, cluster: Cluster, environment: EnvironmentFacts | None, ledger: LedgerFacts | None = None) -> float | None
```

Dollars one step (of an eval: the eval) is estimated to cost on the run's metered parts, at most
(`spend_of`); none where it cannot be estimated.

### `Finding`

*class* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
class Finding
```

One thing wrong with a run's settings: the rule, the key it is about (the field a form marks), and why.

| Field | Type | Default | Description |
|---|---|---|---|
| `rule` | `str` | required |  |
| `key` | `str` | required |  |
| `reason` | `str` | required |  |
| `refuses` | `bool` | `True` | False: the run may go (it waits, or something could not be known); the finding is a note. |

### `LedgerFacts`

*class* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
class LedgerFacts
```

What the ledger and the live cluster say, asked beforehand.

| Field | Type | Default | Description |
|---|---|---|---|
| `checkpoints` | `Mapping[str, CheckpointFacts]` | `field(default_factory=dict[str, CheckpointFacts])` | Every checkpoint reference the settings name, looked up (one not here does not exist). |
| `suites` | `Mapping[str, SuiteFacts]` | `field(default_factory=dict[str, SuiteFacts])` |  |
| `names_taken` | `frozenset[str]` | `frozenset()` |  |
| `gpus` | `float \| None` | `None` | GPUs the cluster has in all (none: not known). |
| `free` | `Resources \| None` | `None` | What the cluster has free now (none: not known). |
| `step_seconds` | `float \| None` | `None` | How long a step of the run's trainer and model took here lately (none: not known): what its pods' hours per step are reckoned from. |

### `refusals`

*function* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
def refusals(findings: list[Finding]) -> list[Finding]
```

The findings that refuse the run.

### `renderers_of`

*function* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
def renderers_of(settings: RunSettings, cluster: Cluster, channel: str) -> list[str]
```

The renderers that render a channel's model, as `module:name` (`rollout_train.recorder.renderers.renderers_for`):
over the model its provider says a quantized model was made from, where none renders the model itself.

### `Rule`

*class* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
class Rule
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `refuses` | `str` | required | When it refuses a run, in words. |

### `RULES`

*constant* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
RULES: tuple[Rule, ...] = (Rule('settings', 'a key the kind does not take, a wrong type or range, a required key missing, contradictions'), Rule('providers', "the trainer or a channel's provider is not offered, or a hosted API shares a channel"), Rule('auth', 'a provider reached with no auth away from this machine'), Rule('capabilities', "the trained channel's provider is a hosted API (no exact tokens or behaviour logprobs, whatever the objective), is not token-exact (a policy gradient), or lacks sampled logprobs and honoured sampling (an importance correction)"), Rule('bridge', "no bridge from the checkpoint's format to what the provider loads"), Rule('weights', 'a trainer that makes the other kind of weights than the run trains; a LoRA on a provider without adapters, full weights on one without full reload'), Rule('models', 'a model not offered, or not the one trained'), Rule('renderer', 'a channel sampling tokens whose model no renderer renders, that several do with none said, or a renderer said that says it renders other models'), Rule('rank', "the adapter's rank, as the provider sees it, above its highest"), Rule('segment', 'segments longer than the trainer or the context takes'), Rule('memory', "a trainer whose estimate of what each of its GPUs holds (weights, gradients, optimizer state and activations, sharded over its GPUs) is more than a GPU's memory"), Rule('start', 'the start does not exist, was released, or is in a format the trainer cannot start from'), Rule('objective', 'a component its family does not accept, a combination that means nothing, a family the trainer or the kind of run does not take, a reference, an entropy or logprobs of tokens not sampled that the trainer cannot give'), Rule('evals', 'a suite that does not exist, or whose environment is not offered'), Rule('distillation', 'no teacher for a route or for the environment played, a teacher without the logprobs distillation reads or whose logprobs are unchecked, or of another renderer family'), Rule('environment', 'not offered, does not load, needs sandboxes or tool sets the cluster lacks, or, on Kubernetes, sandboxes whose pool is not served from pods of its own'), Rule('capacity', "more than the cluster schedules for one run ([capacity]), or more GPUs than it has, counting the run's scheduled parts"), Rule('spend', "a training run's spend limit below one step's estimated cost"), Rule('name', 'not a name, or taken'))
```

Every rule `check` applies, in the order it reports them.

### `serves`

*function* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
def serves(provider: InferenceProvider, weights: str) -> str | None
```

Why `provider` cannot serve a run's checkpoints of `weights` (`lora`: adapters by name; `full`: full weights
reloaded in place), if it cannot.

### `Spend`

*class* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
class Spend
```

A run's estimated spend on its metered parts, at most. For a training run, one step's (`per` is `step`): every
token trained times the trainer's cost for the model, and the sampled and prompt tokens times the dearest metered
provider's costs to sample and read them, uncached. For an eval, the whole eval's (`per` is `eval`): every
episode of the suite's starts, each turn's thinking and answer budgets sampled and its prompt read at the
dearest metered provider of the channel it plays. Scheduled parts are capacity the run is placed on, not spent.

| Field | Type | Default | Description |
|---|---|---|---|
| `dollars` | `float \| None` | required | None where it cannot be estimated (`why`). |
| `parts` | `Mapping[str, float]` | `field(default_factory=dict[str, float])` | Each metered part's dollars, by provider or trainer. |
| `why` | `str` | `''` | Why it cannot be estimated. |
| `per` | `str` | `'step'` | What it is the spend of: one step of a training run (`step`), or a whole eval (`eval`). |

### `spend_of`

*function* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
def spend_of(settings: RunSettings, cluster: Cluster, environment: EnvironmentFacts | None, ledger: LedgerFacts | None = None) -> Spend
```

A run's estimated spend (`Spend`), or why it cannot be estimated: one step of a training run, or a whole eval
(from the suite's starts in `ledger`); not for other runs; for a training run, not without a trainer; not where the
environment's numbers or the budgets are unknown, or a metered part is priced by the hour. Episodes a group are
the run's `group_size`, else the environment's, else the objective's group size; each turn of an episode is as many
samples as the environment says (`samples_per_turn`: every agent of a team samples each turn).

### `SuiteEntryFacts`

*class* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
class SuiteEntryFacts
```

One entry of the version of a suite the settings name: what an eval of it plays.

| Field | Type | Default | Description |
|---|---|---|---|
| `environment` | `str` | required |  |
| `starts` | `int` | required |  |
| `episodes` | `int` | required | Of each start, unless the eval says another number. |
| `thinking_tokens` | `int \| None` | `None` |  |
| `answer_tokens` | `int \| None` | `None` |  |

### `SuiteFacts`

*class* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
class SuiteFacts
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required |  |
| `newest` | `int` | required | Its newest version's number. |
| `environments` | `frozenset[str]` | `frozenset()` |  |
| `entries` | `tuple[SuiteEntryFacts, ...]` | `()` | The entries of the version the settings name (its newest, where they name none). |

### `weights_of`

*function* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
def weights_of(settings: RunSettings, cluster: Cluster) -> str | None
```

What a run trains: its `weights`, else what its trainer makes (`lora` or `full`); none where neither is known.

### `with_renderers`

*function* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
def with_renderers(settings: RunSettings, cluster: Cluster) -> RunSettings
```

A run's settings with each channel's renderer said where it names a model and no renderer, and exactly one
renderer renders that model (`renderers_of`).

### `with_weights`

*function* · `libraries/rollout-train/src/rollout_train/validation.py`

```python
def with_weights(settings: RunSettings, cluster: Cluster) -> RunSettings
```

A training run's settings with what it trains said (`weights_of`), as its start records them.

## `rollout_train.memory`

What a trainer needs of each GPU's memory, from the model's files and its GPUs.

### `ALLOWANCE_GIB`

*constant* · `libraries/rollout-train/src/rollout_train/memory.py`

```python
ALLOWANCE_GIB = 3.0
```

Each GPU's memory beyond what the estimate counts: the CUDA context, the collectives' buffers, fragmentation.

### `GPU_MEMORY_GIB`

*constant* · `libraries/rollout-train/src/rollout_train/memory.py`

```python
GPU_MEMORY_GIB: Mapping[str, float] = {'NVIDIA B200': 180, 'NVIDIA H200': 141, 'NVIDIA H200 NVL': 141, 'NVIDIA H100 80GB HBM3': 80, 'NVIDIA H100 NVL': 94, 'NVIDIA H100 PCIe': 80, 'NVIDIA A100-SXM4-80GB': 80, 'NVIDIA A100 80GB PCIe': 80, 'NVIDIA RTX PRO 6000 Blackwell Server Edition': 96, 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition': 96, 'NVIDIA L40S': 48, 'NVIDIA L40': 48, 'NVIDIA RTX 6000 Ada Generation': 48, 'NVIDIA RTX A6000': 48, 'NVIDIA A40': 48, 'NVIDIA GeForce RTX 5090': 32, 'NVIDIA GeForce RTX 4090': 24, 'NVIDIA L4': 24}
```

Each GPU's memory by RunPod's GPU type id (a type not here says it in its id, `80GB`, or is not known).

### `gpu_memory_gib`

*function* · `libraries/rollout-train/src/rollout_train/memory.py`

```python
def gpu_memory_gib(gpu_types: tuple[str, ...]) -> float | None
```

The least memory of any of RunPod's GPU types (`GPU_MEMORY_GIB`, or the `80GB` its id says); none where one is
not known.

### `holds_whole_base`

*function* · `libraries/rollout-train/src/rollout_train/memory.py`

```python
def holds_whole_base(file_bytes: float, gpu_gib: float | None) -> bool
```

Whether each GPU holds an adapter's whole frozen model where the settings do not say (`whole_base`): where the
model's files take at most half of a GPU's memory (`gpu_gib`, as `gpu_memory_gib` gives it; none: not known). The
estimate decides by it, and so does the trainer, from its GPUs' names.

### `model_facts`

*function* · `libraries/rollout-train/src/rollout_train/memory.py`

```python
def model_facts(model: str, *, environ: Mapping[str, str] | None = None, patience: float = 5.0) -> ModelFacts | None
```

What a model's files say of its size: read from its directory or the Hugging Face cache, else asked of the
Hugging Face Hub, for at most `patience` seconds (`config.json` and the safetensors index; `HF_TOKEN` for a gated
model; never with `HF_HUB_OFFLINE`); none where neither says. It blocks: call it in a thread.

### `ModelFacts`

*class* · `libraries/rollout-train/src/rollout_train/memory.py`

```python
class ModelFacts
```

What a model's files say of its size.

| Field | Type | Default | Description |
|---|---|---|---|
| `file_bytes` | `int` | required | Its weights' files, in all (a quantized model's are its quantized weights). |
| `parameters` | `float \| None` | required | Its parameters (none for a quantized model, whose files do not say them). |
| `hidden` | `int` | required |  |
| `layers` | `int` | required |  |
| `vocabulary` | `int` | required |  |
| `tied` | `bool` | required | Whether its output layer is its token embeddings. |
| `linear_state` | `int` | `0` | The values of a linear-attention layer's state (value heads times key width times value width); 0 for a model without linear attention. |

### `SEGMENT_TOKENS`

*constant* · `libraries/rollout-train/src/rollout_train/memory.py`

```python
SEGMENT_TOKENS = 8192
```

The segment an estimate allows activations for where the trainer says no longest one (`segment_tokens`).

### `trainer_memory`

*function* · `libraries/rollout-train/src/rollout_train/memory.py`

```python
def trainer_memory(model: ModelFacts, *, weights: str, gpus: int, rank: int = 32, segment_tokens: int | None = None, frozen_reference: bool = False, whole_base: bool | None = None, gpu_gib: float | None = None) -> TrainerMemory
```

What a trainer of `weights` (`lora` or `full`) over `model` on `gpus` GPUs needs of each
(`rollout_train.memory`). `whole_base` is the adapter's setting (none: whole where the model takes at most half of
`gpu_gib`: `holds_whole_base`).

### `TrainerMemory`

*class* · `libraries/rollout-train/src/rollout_train/memory.py`

```python
class TrainerMemory
```

What a trainer needs of each GPU, in GiB, by part (`rollout_train.memory`).

| Field | Type | Default | Description |
|---|---|---|---|
| `gpus` | `int` | required |  |
| `weights` | `float` | required |  |
| `gradients` | `float` | required |  |
| `optimizer` | `float` | required |  |
| `reference` | `float` | required |  |
| `activations` | `float` | required |  |
| `allowance` | `float` | `ALLOWANCE_GIB` |  |
| `whole_base` | `bool` | `False` | For an adapter on several GPUs: whether the estimate has each hold the whole frozen model. |

**Methods**

- `@property def total(self) -> float`
- `def said(self) -> str` — In words: `52 GiB a GPU (weights 18, gradients 9, ...)`.

## `rollout_train.slots`

A program's model slots bound to a run's channels, and the bindings a run may not make.

### `bound`

*function* · `libraries/rollout-train/src/rollout_train/slots.py`

```python
def bound(settings: RunSettings, declared: Declared) -> dict[str, str]
```

The channel each declared slot samples: the one `slots.SLOT` names, else, for a trained slot, the subject
channel (an untrained slot the settings do not bind is left out: `problems` refuses it).

### `Declared`

*class* · `libraries/rollout-train/src/rollout_train/slots.py`

```python
class Declared
```

The slots a program declares: their names, those that are not trained, and those that judge.

| Field | Type | Default | Description |
|---|---|---|---|
| `names` | `frozenset[str]` | required |  |
| `untrained` | `frozenset[str]` | `frozenset()` |  |
| `judges` | `frozenset[str]` | `frozenset()` |  |

**Methods**

- `@classmethod def of(cls, slots: Mapping[str, ModelSlot]) -> 'Declared'`

### `problems` {#rollout_trainslotsproblems}

*function* · `libraries/rollout-train/src/rollout_train/slots.py`

```python
def problems(settings: RunSettings, declared: Declared, own: Collection[str] | None = None) -> list[tuple[str, str]]
```

What a run's bindings of the declared slots break, as `(key, reason)`: an untrained slot left unbound; a channel
a slot is bound to with no provider or model (`settings.providers` where the run names providers); a judge bound
to a channel serving the run's own checkpoints (`own`, by default `serving`) without `self_judging`.

### `serving`

*function* · `libraries/rollout-train/src/rollout_train/slots.py`

```python
def serving(settings: RunSettings) -> set[str]
```

The channels that serve the run's own checkpoints: the trained channel, and those that follow it (directly or
through others).

### `subject`

*function* · `libraries/rollout-train/src/rollout_train/slots.py`

```python
def subject(settings: RunSettings) -> str
```

The channel a trained slot the settings do not bind samples: the trained channel.

## `rollout_train.testing`

Test doubles: a scripted engine and a readable token format.

### `admitted`

*function* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
async def admitted(endpoints: GatewayEndpoints, run_id: str, run: str = 'train') -> Attempt
```

Admit a run that no episode runner plays (one a test starts on a runner itself), under a fence of its own.

### `Characters`

*class* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
class Characters
```

A tokenizer of one token per character.

**Methods**

- `def encode(self, text: str, add_special_tokens: bool = False) -> list[int]`
- `def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str`

### `gateway_endpoints`

*function* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
def gateway_endpoints(*channels: Channel, ledger: Ledger, blobs: Blobs, url: str | None = None, routes: Routes | None = None, hooks: Sequence[RunHooks] = (), models: Mapping[str, str] | None = None) -> GatewayEndpoints
```

Endpoints over a gateway in this process that samples `channels` (`models` names each one's base model, by
channel) and those `routes` route, recording in `ledger` and `blobs`, with the keys of `SECRETS`; `url` is where it
is served to harnesses, if it is, and `hooks` are told of the samples harnesses ask for there.

### `keyring`

*function* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
def keyring() -> Keyring
```

The keys of `SECRETS`.

### `LEDGER_TOKEN`

*constant* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
LEDGER_TOKEN = 'the-platform-token-of-a-test-ledger'
```

The platform's token of a test's ledger service (`served_ledger`).

### `plain_channel`

*function* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
def plain_channel(script: Sequence[tuple[str, str]] = (), *, name: str = 'policy', **options: Any) -> Channel
```

A channel over a scripted engine in the plain format; `always=` repeats a script for ever.

### `plain_renderer`

*function* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
def plain_renderer(model: str) -> Renderer
```

### `PlainRenderer`

*class* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
class PlainRenderer
```

A token format for tests: each message is `role: text` on a line, a tool call is `call NAME {json}`, and a
turn ends with the line. A reply renders back exactly as it was sampled, so a conversation that only grows is
one segment.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` |  | `'plain'` |  |
| `thinking` |  | `None` |  |

**Methods**

- `def render(self, messages: Sequence[Message], tools: Sequence[ToolSpecification]) -> list[int]`
- `def encode(self, text: str) -> list[int]`
- `def decode(self, tokens: Sequence[int]) -> str`
- `def stop_token_ids(self) -> list[int]`
- `def thinking_end_token_ids(self) -> list[int]`
- `def parse(self, completion: Sequence[int], tools: Sequence[ToolSpecification]) -> Message`

### `Policy`

*class* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
class Policy
```

Channels whose engines are in this process, as a test's training loop and its runners see them: `publish` serves
new weights on one (what a loop is given to publish with), and `gateway` is what a runner samples through.

**Methods**

- `def __init__(self, *channels: Channel) -> None`
- `async def publish(self, channel: str, adapter: str, files: Callable[[], Awaitable[str]], version: int | None = None, *, full: bool = False) -> int`
- `def gateway(self, ledger: Ledger, blobs: Blobs) -> GatewayEndpoints` — A gateway in this process over the channels, recording in `ledger` and `blobs`: one for each place the
  ledger is, so that a runner made first and an episode runner made after it over the same place share it (the
  runs it admits are the runs the runner plays).

### `sample_request`

*function* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
def sample_request(messages: list[Message], effect_id: str = 'r_1:0:0', *, session_id: str = 'r_1/ada', tools: Sequence[ToolSpecification] = ()) -> SampleRequest
```

### `scripted_engine`

*function* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
def scripted_engine(model: str, **options: Any) -> ScriptedEngine
```

An engine whose policy says yes and no in turn; `fails=true` makes one that cannot start, and `gate` (a
directory) holds its loads of full weights back until a test lets each go (`ScriptedEngine`).

### `scripted_top`

*function* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
def scripted_top(tokens: Sequence[int], top: int, logprob: float = -0.25) -> tuple[list[list[int]], list[list[float]]]
```

The `top` most likely tokens a scripted engine gives at each position of `tokens`: the token there (at
`logprob`) and those after it, each a nat less likely than the one before (none when `top` is 0).

### `ScriptedEngine`

*class* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
class ScriptedEngine
```

Answers each generate with the next scripted (text, finish reason), or with `always` once the script is
spent; logprobs are -0.5 per token. Scores each token at -0.25. Keeps what it was asked and told. With `gate` (a
directory), its `N`th load of full weights (from 1) writes `loading-N` there and waits until a file `N` is there
too.

| Field | Type | Default | Description |
|---|---|---|---|
| `max_model_len` |  | `32768` |  |
| `processes` | `Sequence[int]` | `()` |  |

**Methods**

- `def __init__(self, tokenizer: Tokenizer, script: Sequence[tuple[str, str]] = (), *, always: Sequence[tuple[str, str]] = (), gate: Path | None = None) -> None`
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, top: int = 0) -> Generation`
- `async def score(self, tokens: Sequence[int], *, start: int, end: int | None = None, top: int = 0, adapter: str | None) -> Scores` — Each token scored at -0.25, its `top` most likely being itself and the tokens after it (`scripted_top`).
- `async def load_adapter(self, name: str, path: str) -> None`
- `async def remove_adapter(self, name: str) -> None`
- `async def load_weights(self, path: str) -> None`
- `async def sleep(self) -> None`
- `async def wake(self) -> None`
- `def close(self) -> None`

### `ScriptedTrainer`

*class* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
class ScriptedTrainer
```

A trainer that trains nothing: each step writes an adapter's files that say how many segments it was given (as
PEFT's are named, so a bridge takes them for an adapter), and the trainer's state beside them. What a cluster
config's trainer names as its `implementation` in tests.

| Field | Type | Default | Description |
|---|---|---|---|
| `weights` |  | `'lora'` |  |

**Methods**

- `def __init__(self, model: str, *, segment_tokens: int | None = None, segments_per_step: int | None = None, **settings: Any) -> None`
- `async def step(self, batch: Sequence[Any], *, seed: int, parent: Any, into: Path) -> Any`

### `SECRETS`

*constant* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
SECRETS = [('k2', 'a-newer-secret-of-thirty-two-bytes!!'), ('k1', 'an-older-secret-of-thirty-two-bytes!')]
```

What a test's gateway (`gateway_endpoints`) signs keys with (the first) and takes keys signed with (each).

### `served_ledger`

*function* · `libraries/rollout-train/src/rollout_train/testing.py`

```python
def served_ledger(ledger: Ledger, *, token: str = LEDGER_TOKEN, secret: str = LEDGER_TOKEN, **service: Any) -> Any
```

`ledger` through the ledger service, served in this process (no socket): an `HttpLedger` that sends `token`
to a service whose platform token is `secret`. `service` goes to the service
(`rollout_train.ledger_service.app`).

## `rollout_vllm`

An engine on vLLM.

### `VllmEngine`

*class* · `implementations/rollout-vllm/src/rollout_vllm/engine.py`

```python
class VllmEngine
```

**Methods**

- `def __init__(self, model: str, *, gpu_memory_utilization: float = 0.72, max_model_len: int = 8192, max_num_seqs: int = 32, max_num_batched_tokens: int = 4096, max_lora_rank: int = 32, max_loras: int = 2, language_model_only: bool = True, speculative: Mapping[str, Any] | None = None, quantization: str | None = None, seed: int = 0, max_logprobs: int = 20) -> None`
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, top: int = 0) -> Generation`
- `async def score(self, tokens: Sequence[int], *, start: int, end: int | None = None, top: int = 0, adapter: str | None) -> Scores` — The logprobs the model (or `adapter`) gives the tokens at positions `start` to `end` of `tokens`, each given
  those before it, and the `top` most likely tokens at each: vLLM's prompt logprobs of `tokens` up to `end`, with
  one token generated (vLLM generates at least one) and dropped. The scores are of the model's own distribution
  (prompt logprobs skip temperature). The sequence must leave room for that token (`max_model_len`). vLLM
  computes the logits of every position before `end`, 1,024 positions at a time, so scoring a long sequence takes
  about 0.6 GiB of the GPU beyond the engine's own share (Qwen3's vocabulary of 152K), however long it is.
- `async def load_adapter(self, name: str, path: str) -> None` — Register a LoRA adapter (a PEFT directory) under `name`; samples name it to use it.
- `async def remove_adapter(self, name: str) -> None`
- `async def load_weights(self, path: str) -> None` — Serve the full weights in `path` (a checkpoint's files, in the model's own layout) in place of the ones
  held, from now on: they are read into the model as it is, and read again from there on waking. (vLLM warns
  that `ParallelLMHead` failed to load where the output layer is tied to the embeddings: it shares them, and
  serves the new ones.)
- `async def sleep(self) -> None` — Free the GPU: the cache is discarded and the weights dropped (they are read again on waking).
- `async def wake(self) -> None`
- `@property def processes(self) -> list[int]`
- `def close(self) -> None`

## `rollout_lora`

A trainer for 4-bit checkpoints with LoRA.

### `FullTrainer`

*class* · `implementations/rollout-lora/src/rollout_lora/trainer.py`

```python
class FullTrainer(LoraTrainer)
```

Trains every weight of a text model (`rollout_lora.full`): a step starts from its parent's weights (the model's
own for the first) and the optimizer's state, and leaves the new ones where it is told. The weights, gradients and
optimizer's state are sharded over its GPUs (on one too), and a step leaves the weights in bfloat16 (what engines
serve) and the full state (the float32 weights and the optimizer's) every `state_every` steps. `settings` are
`LoraSettings`' fields; `rank` is not used. It holds a reference (a frozen copy of the model) only when asked
(`frozen_reference`).

| Field | Type | Default | Description |
|---|---|---|---|
| `weights` |  | `'full'` |  |

### `LoraSettings`

*class* · `implementations/rollout-lora/src/rollout_lora/settings.py`

```python
class LoraSettings(StepSettings)
```

The settings of `LoraTrainer` and `FullTrainer`: a step's, and the adapter's scaling, which is twice its rank
(`alpha`).

| Field | Type | Default | Description |
|---|---|---|---|
| `frozen_reference` | `bool` | `False` | For the full-weight trainer: hold a frozen copy of the model trained over (in bfloat16, beside the policy), the reference an objective may read. An adapter's reference is the model with the adapter switched off. |
| `state_every` | `int` | `1` | How often a trainer whose processes are kept between steps (one with its GPUs to itself) writes its full state: the optimizer's, and full weights in float32. Every step writes the weights (an adapter in float32, full weights as a bfloat16 serving copy) whatever this says. 1: every step, so that any trainer goes on from any checkpoint as the trainer that made it would. More: the steps between leave the full state out, which saves writing about 12 bytes a weight for full weights, and a step from one of them goes on only from the processes that hold it. Once those are gone (after a failed step, a restart, or another run taking a training pod) such a step fails (`StepFailed`) rather than go on from less than its parent was: a run started from the newest checkpoint with its full state goes on. A trainer whose processes end after each step (one beside an engine) writes it every step. |
| `whole_base` | `bool \| None` | `None` | For an adapter on several GPUs: whether each GPU holds the whole frozen model, gathered once (true: no gathering for each segment, for a model that fits one GPU beside its activations), or a share of it, each layer gathered as it computes (false: a model too large for one GPU). None: whole where the model's files take at most half of a GPU's memory (`rollout_train.memory.holds_whole_base`, which the memory estimate decides by too). |

**Methods**

- `@property def alpha(self) -> float`

### `LoraTrainer`

*class* · `implementations/rollout-lora/src/rollout_lora/trainer.py`

```python
class LoraTrainer
```

Trains a LoRA adapter over `model`'s checkpoint, one step at a time. `settings` are `LoraSettings`' fields (its
`objective` among them); those in `CHANGEABLE`, and the changeable components of its objective, it takes between
steps (`rollout_train.trainer.Changeable`). Its reference is the model with the adapter switched off.

`gpus` is how many GPUs it steps on (by default those it is given: `CUDA_VISIBLE_DEVICES`, which Ray sets for a
trainer's actor, else the machine's), a process on each (`rollout_lora.workers`), the policy sharded over them on
more than one (`rollout_lora.sharded`). `colocated` says it shares its GPU with an inference engine, which sleeps
while it steps: then each step's processes end after it, and give the engine back the memory. Otherwise they are
kept between steps: a step from the checkpoint the last one made goes on from what they hold
(`rollout_train.trainer.Resident`). Every step leaves the files a later one starts from (the full state every
`state_every` steps): in `into/state` before it returns, or, told a blob store (`keep_in`), kept there by the
processes after it returns (`kept`), its weights served meanwhile.

| Field | Type | Default | Description |
|---|---|---|---|
| `weights` |  | `'lora'` |  |

**Methods**

- `def __init__(self, model: str, *, gpus: int | None = None, colocated: bool = False, **settings: Any) -> None`
- `@property def objective(self) -> Objective`
- `@property def changeable(self) -> Mapping[str, JsonValue]`
- `@property def holding(self) -> str | None` — What its processes hold between steps (none beside an engine, where they end after each step).
- `def change(self, settings: Mapping[str, JsonValue]) -> None`
- `def keep_in(self, blobs: Mapping[str, JsonValue]) -> None` — Have the processes keep each step's full state in the blob store at this location after the step returns
  (where they are kept between steps: beside an engine they end after each step, and write it before).
- `async def kept(self, into: str) -> Manifest` — What the processes kept of the full state of the step that wrote into `into`, once they all have.
- `async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step`
- `def close(self) -> None` — End its processes, and what they hold (the next step starts them again).

## `rollout_objectives.settings`

A policy step's settings, which the LoRA, full-weight and Tinker trainers take.

### `CHANGEABLE`

*constant* · `implementations/rollout-objectives/src/rollout_objectives/settings.py`

```python
CHANGEABLE = ('learning_rate', 'tokens_per_step', 'max_kl', 'max_gradient_norm')
```

The settings a trainer takes between steps: each step reads them afresh, and none changes what its weights are or
what a step can hold. Beside them, the components of its objective that may change (`objective.kl.coefficient`).

### `OBJECTIVE`

*constant* · `implementations/rollout-objectives/src/rollout_objectives/settings.py`

```python
OBJECTIVE = 'objective.'
```

The start of a changeable setting that is a component of the objective, as the run's settings name it.

### `StepSettings`

*class* · `implementations/rollout-objectives/src/rollout_objectives/settings.py`

```python
class StepSettings
```

A policy step's settings (`rollout_objectives.step`), whichever trainer takes it.

| Field | Type | Default | Description |
|---|---|---|---|
| `rank` | `int` | `32` | Of the adapter. |
| `learning_rate` | `float` | `5e-05` |  |
| `tokens_per_step` | `int` | `4096` | Sampled tokens per optimizer step (gradients accumulate over segments until then). Adam moves a weight by at most the learning rate a step, so how far an update goes is set by how many steps its tokens make. |
| `max_kl` | `float \| None` | `0.02` | Stop the pass when a minibatch, before its step, finds the policy this far from where the step began (in nats per token, estimated on the sampled tokens). A likelihood step does not stop. |
| `max_gradient_norm` | `float` | `1.0` |  |
| `segment_tokens` | `int \| None` | `None` | The longest segment a step can hold (None: any). Longer ones are left out and counted (`segments_too_long`): one too long would end or stall the whole step. Leaving segments out biases training, so whoever serves the policy takes this as the longest turn to sample; the count says whether that held. |
| `segments_per_step` | `int \| None` | `None` | How many segments a step can afford (None: any number). |
| `pack_tokens` | `int \| None` | `None` | The most tokens one forward and backward pass runs: a step packs its segments into rows of up to this many (`rollout_objectives.packing`). None: `segment_tokens`, so that a pack takes about the memory the longest segment would alone (a pack's activations are those of a segment as long as its row, whatever prefixes its segments share), or 8,192 where that is none (what the trainer's memory estimate allows for, `rollout_train.memory.SEGMENT_TOKENS`). A segment longer than it has a pack of its own. |
| `share_prefixes` | `bool` | `True` | Whether segments of a pack that start with the same tokens share them: the prefix is run once, and each segment's rest after it. |
| `passes` | `int` | `1` | Passes a step takes over its segments, each shuffled anew and cut into minibatches of its own: a small batch makes more optimizer updates (a supervised step on a small dataset, say). |
| `warmup_updates` | `int` | `0` | When a step's optimizer starts afresh (no state to go on from), its rate rises linearly over its first this many updates, from `learning_rate / warmup_updates` to `learning_rate`: a fresh Adam's first update moves every weight by about the full rate. A step that goes on from an optimizer's state is not warmed up. |
| `objective` | `Objective \| str \| Mapping[str, Any]` | `DEFAULT` | The objective (`rollout_train.objectives`): an `Objective`, a preset's name, or a table of `preset` and component overrides (`rollout_train.objectives.objective_of`). The run's `objective.*` settings say it; `loss` is it, resolved. |
| `ratio` | `InitVar[str]` | `_UNSAID` |  |
| `clip_low` | `InitVar[float]` | `_UNSAID` |  |
| `clip_high` | `InitVar[float]` | `_UNSAID` |  |
| `segment_clip_low` | `InitVar[float]` | `_UNSAID` |  |
| `segment_clip_high` | `InitVar[float]` | `_UNSAID` |  |
| `truncate` | `InitVar[float \| None]` | `_UNSAID` | The objective's components by a trainer's own names (`rollout_train.objectives.from_trainer_settings`): `ratio` (`token`, `segment`), the clip of a token ratio (`clip_low`, `clip_high`) or a segment ratio (`segment_clip_low`, `segment_clip_high`), the importance weight's cap (`truncate`; none: the weight untruncated). |

**Methods**

- `@property def loss(self) -> Objective` — The objective a step takes, resolved.
- `def rate(self, update: int, *, fresh: bool) -> float` — The learning rate of a step's `update`-th optimizer update (from 0): warmed up if its optimizer is
  `fresh`.
- `def changeable(self) -> dict[str, JsonValue]` — The settings of `CHANGEABLE` and the changeable components of its objective's family (by their run
  settings' keys, `objective.kl.coefficient`), with their values (what `rollout_train.trainer.Changeable`
  says).
- `def changed(self, changes: Mapping[str, JsonValue]) -> Self` — These settings with `changes`, each one of `changeable` (else `ValueError`), checked as any settings are.

## `rollout_objectives.terms`

An objective's loss composed from its components, in torch.

### `importance_weight`

*function* · `implementations/rollout-objectives/src/rollout_objectives/terms.py`

```python
def importance_weight(objective: Objective, old: torch.Tensor, behavior: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None]
```

Each sampled token's importance weight as `importance.correction` makes it (a constant), and the weight before
truncation or masking; none for no correction. A mask (masked importance sampling; NeMo's `icepop`) zeroes a token
whose weight is outside `floor` .. `cap`, keeping the weight of those within.

### `kl_estimate`

*function* · `implementations/rollout-objectives/src/rollout_objectives/terms.py`

```python
def kl_estimate(estimator: str, logprobs: torch.Tensor, target: torch.Tensor) -> torch.Tensor
```

Each sampled token's estimate of KL(policy || target), from `log_r = target - logprobs` (Schulman's k1, k2,
k3).

### `labelled`

*function* · `implementations/rollout-objectives/src/rollout_objectives/terms.py`

```python
def labelled(objective: Objective, examples: Sequence[tuple[Scored, bool]]) -> list[Terms]
```

KTO's loss of each labelled example (desirable or not), against the reference point `z`: the mean of the
examples' log ratios, no less than 0, with no gradient (an estimate of the policy's KL to the reference).

### `likelihood`

*function* · `implementations/rollout-objectives/src/rollout_objectives/terms.py`

```python
def likelihood(objective: Objective, logprobs: torch.Tensor, advantage: float) -> Terms
```

One segment's likelihood loss: its sampled tokens' log-likelihood, weighted by its advantage (imitation: what
was sampled is what to do), reading neither `old` nor `behavior`.

### `moved_kl`

*function* · `implementations/rollout-objectives/src/rollout_objectives/terms.py`

```python
def moved_kl(old: torch.Tensor, logprobs: torch.Tensor) -> torch.Tensor
```

Each sampled token's estimate of KL(old || now), how far the step has moved the policy (no gradient): Schulman's
k3 of `log_r = logprobs - old`, `(r - 1) - log r`. On tokens sampled where the step began its mean is the KL's,
and no token's estimate is below 0. The k1 estimate, `old - logprobs`, has the same mean but either sign: where an
update makes every sampled token likelier (every advantage positive, as REINFORCE without a baseline, or a
distillation whose teacher is surer than the student), it reads near 0 or below however far the policy moved.

### `pair`

*function* · `implementations/rollout-objectives/src/rollout_objectives/terms.py`

```python
def pair(objective: Objective, chosen: Scored, rejected: Scored) -> Terms
```

A pair's preference loss (`preference.loss`), and a likelihood term on its chosen side
(`likelihood.coefficient`, ORPO's).

### `policy_gradient`

*function* · `implementations/rollout-objectives/src/rollout_objectives/terms.py`

```python
def policy_gradient(objective: Objective, logprobs: torch.Tensor, advantage: float, old: torch.Tensor, behavior: torch.Tensor | None = None, reference: torch.Tensor | None = None, entropy: torch.Tensor | None = None) -> Terms
```

One segment's policy-gradient loss, from its sampled tokens' logprobs now (with gradient), at the step's start
(`old`), when they were sampled (`behavior`, for an importance correction), under the reference (for a KL to it)
and each position's entropy (for an entropy bonus).

### `reduced`

*function* · `implementations/rollout-objectives/src/rollout_objectives/terms.py`

```python
def reduced(objective: Objective, per_token: torch.Tensor) -> torch.Tensor
```

A segment's per-token losses as its part of the minibatch's (`aggregate`).

### `Scored`

*class* · `implementations/rollout-objectives/src/rollout_objectives/terms.py`

```python
class Scored
```

One side of a preference item: each of its segments' sampled tokens' logprobs now, and the reference's.

| Field | Type | Default | Description |
|---|---|---|---|
| `logprobs` | `Sequence[torch.Tensor]` | required |  |
| `reference` | `Sequence[torch.Tensor] \| None` | `None` |  |

**Methods**

- `def tokens(self) -> int`
- `def likelihood(self, objective: Objective) -> torch.Tensor` — Its log-likelihood: the sum of its tokens' logprobs, or their mean (`length_normalized`).
- `def rho(self, objective: Objective) -> torch.Tensor` — Its log-likelihood ratio to the reference, or its log-likelihood where the loss has none.

### `SUMS`

*constant* · `implementations/rollout-objectives/src/rollout_objectives/terms.py`

```python
SUMS: tuple[str, ...] = ('loss', 'units', 'segments', *TALLIED)
```

What a minibatch's `Terms` add up to, for its statistics and the step's.

### `TALLIED`

*constant* · `implementations/rollout-objectives/src/rollout_objectives/terms.py`

```python
TALLIED = ('clipped', 'truncated', 'tokens', 'ratio', 'weight', 'moved', 'kl', 'entropy', 'items', 'pairs', 'accurate', 'margin', 'chosen', 'rejected', 'distilled', 'scored', 'gap', 'divergence', 'advantage_clipped')
```

The counts and sums of `Terms` a minibatch adds up.

### `tally`

*function* · `implementations/rollout-objectives/src/rollout_objectives/terms.py`

```python
def tally(sums: dict[str, float], found: Terms, objective: Objective, *, segments: float = 1.0) -> None
```

Add one segment's `found` terms (or one preference item's, of `segments` segments) to `sums` (keyed by
`SUMS`).

### `Terms`

*class* · `implementations/rollout-objectives/src/rollout_objectives/terms.py`

```python
class Terms
```

One segment's part of a minibatch's loss (`loss`, which the trainer divides by the minibatch's units), or one
preference item's; the rest are counts and sums, for the step's statistics.

| Field | Type | Default | Description |
|---|---|---|---|
| `loss` | `torch.Tensor` | required |  |
| `tokens` | `float` | required |  |
| `clipped` | `float` | `0.0` | Tokens whose ratio was clipped. |
| `truncated` | `float` | `0.0` | Tokens whose importance weight was truncated, or that a mask dropped. |
| `ratio` | `float` | `0.0` |  |
| `weight` | `float` | `0.0` |  |
| `moved` | `float` | `0.0` | The sum of `moved_kl` over the tokens: an estimate of KL(old \|\| now) on the sampled tokens, times their number. |
| `kl` | `float` | `0.0` | The sum of the KL penalty's estimate over the tokens. |
| `entropy` | `float` | `0.0` |  |
| `items` | `float` | `0.0` | Preference items: pairs or examples. |
| `pairs` | `float` | `0.0` |  |
| `accurate` | `float` | `0.0` | Of the items, those whose chosen side's log ratio is above the rejected's (an example's above the reference point if desirable, below it if not). |
| `margin` | `float` | `0.0` | The sum of each pair's `rho_chosen - rho_rejected`, and of each example's distance from the reference point on its label's side (`rho - z` if desirable, `z - rho` if not). |
| `chosen` | `float` | `0.0` |  |
| `rejected` | `float` | `0.0` | Sums of each pair's `rho_chosen` and `rho_rejected`. |
| `distilled` | `float` | `0.0` | Sampled tokens of distilled segments. |
| `scored` | `float` | `0.0` | Of those, the tokens a teacher scored. |
| `gap` | `float` | `0.0` | The sum, over the scored tokens, of the policy's logprob now less the teacher's: an estimate of KL(policy \|\| teacher) on the sampled tokens, times their number, where the policy sampled them. |
| `divergence` | `float` | `0.0` | The sum of the top-k divergence over the scored tokens. |
| `advantage_clipped` | `float` | `0.0` | Scored tokens whose distillation advantage was clipped. |

### `terms`

*function* · `implementations/rollout-objectives/src/rollout_objectives/terms.py`

```python
def terms(objective: Objective, logprobs: torch.Tensor, advantage: float, old: torch.Tensor | None = None, behavior: torch.Tensor | None = None, reference: torch.Tensor | None = None, entropy: torch.Tensor | None = None) -> Terms
```

One weighted segment's loss under `objective`: a policy gradient's or a likelihood's.

### `units`

*function* · `implementations/rollout-objectives/src/rollout_objectives/terms.py`

```python
def units(objective: Objective, tokens: int) -> float
```

What a segment of `tokens` sampled tokens counts for in its minibatch's mean (a preference item counts 1).

## `rollout_objectives.step`

A step over a batch on a local policy, its plan of minibatches, and its statistics.

### `line`

*function* · `implementations/rollout-objectives/src/rollout_objectives/step.py`

```python
def line(sums: Mapping[str, float], units: float, rate: float) -> dict[str, float]
```

What a minibatch that was stepped on did (its `SUMS`, over `units`, stepped at `rate`), as `MINIBATCHES` keeps
it.

### `metrics`

*function* · `implementations/rollout-objectives/src/rollout_objectives/step.py`

```python
def metrics(totals: Mapping[str, float], starts: Sequence[tuple[torch.Tensor, torch.Tensor]], *, plan: Plan, given: int, moved: float, updates: int, settings: StepSettings, fresh: bool, stopped: bool, start_seconds: float) -> dict[str, float]
```

A step's metrics: from the `SUMS` of the minibatches it stepped on (`totals`), each segment's behaviour and
start logprobs (`starts`; a behaviour logprob that is not finite, from a provider without them, is left out of
theirs), the plan of `given` items, how far the last minibatch stepped on found the policy from the step's start
(`moved`), and how many `updates` it made.

### `MINIBATCHES`

*constant* · `implementations/rollout-objectives/src/rollout_objectives/step.py`

```python
MINIBATCHES = 'minibatches.jsonl'
```

In a step's state: what each of its minibatches did, one line each (`line`).

### `minibatches`

*function* · `implementations/rollout-objectives/src/rollout_objectives/step.py`

```python
def minibatches[Each: Item](items: Sequence[Each], tokens_per_step: int) -> list[list[Each]]
```

The items in order, cut where a minibatch has reached `tokens_per_step` sampled tokens. A last minibatch of less
than half that joins the one before: Adam's step is as large for a handful of tokens as for a full minibatch.

### `PackingPolicy`

*class* · `implementations/rollout-objectives/src/rollout_objectives/step.py`

```python
class PackingPolicy(TrainablePolicy, Protocol)
```

A policy that runs a pack of segments in one pass (where `packing`): each segment's sampled tokens' logprobs
as it gives them for the segment alone, with the entropies (`entropy`) or the logprobs of given tokens
(`candidates`, a tensor for each segment) where asked; and under the reference, for an objective that reads it.

| Field | Type | Default | Description |
|---|---|---|---|
| `packing` | `bool` | required |  |

**Methods**

- `def packed(self, pack: Pack, *, entropy: bool = False, candidates: Sequence[torch.Tensor] | None = None) -> list[Scores]`
- `def packed_reference(self, pack: Pack) -> list[torch.Tensor]`

### `Plan` {#rollout_objectivesstepplan}

*class* · `implementations/rollout-objectives/src/rollout_objectives/step.py`

```python
class Plan
```

The items a step trains on, in the order it takes them (those whose segments are all no longer than
`segment_tokens`, with tokens sampled, shuffled by the step's seed), and how many it left out for their length.

| Field | Type | Default | Description |
|---|---|---|---|
| `items` | `list[Item]` | required |  |
| `too_long` | `int` | required |  |
| `shuffled` | `random.Random` | required | What shuffles each pass after the first. |

**Methods**

- `@classmethod def of(cls, items: Sequence[Item], settings: StepSettings, seed: int) -> 'Plan'`
- `@property def segments(self) -> list[Segment]` — Every segment of its items, in order (a segment two items share, once).
- `def minibatches(self, settings: StepSettings) -> list[list[Item]]` — Every pass's minibatches, in order: the first pass takes the items in the plan's order, and each further
  one shuffles them anew.

### `PolicyStep`

*class* · `implementations/rollout-objectives/src/rollout_objectives/step.py`

```python
class PolicyStep
```

| Field | Type | Default | Description |
|---|---|---|---|
| `policy` | `TrainablePolicy` | required |  |
| `settings` | `StepSettings` | `field(default_factory=StepSettings)` |  |
| `fresh` | `bool` | `True` | Whether the optimizer starts afresh (warmed up), or goes on from a state loaded into it. |
| `ranks` | `Ranks` | `field(default_factory=Ranks)` | The processes the step is shared among (one by default: none). |
| `optimizer_given` | `InitVar[torch.optim.Optimizer \| None]` | `None` | An optimizer to go on with (a resident trainer's, from its last step); else a new AdamW. |

**Methods**

- `def step(self, items: Sequence[Item], *, seed: int = 0) -> dict[str, float]` — The logprobs the step starts from, then `passes` passes over the items in shuffled minibatches.

### `preference_terms`

*function* · `implementations/rollout-objectives/src/rollout_objectives/step.py`

```python
def preference_terms(objective: Objective, batch: Sequence[Item], now: Mapping[int, torch.Tensor], reference: Mapping[int, torch.Tensor]) -> list[tuple[Item, Terms]]
```

The preference loss of each item of a minibatch, from each segment's logprobs (`now`, by the segment's `id`)
and the reference's.

### `sampled`

*function* · `implementations/rollout-objectives/src/rollout_objectives/step.py`

```python
def sampled(weighted: Weighted) -> list[int]
```

The positions of the tokens the policy sampled in a weighted segment.

### `SharedPolicy`

*class* · `implementations/rollout-objectives/src/rollout_objectives/step.py`

```python
class SharedPolicy(TrainablePolicy, Protocol)
```

A policy sharded among the processes a step is shared among, whose gradients are added up across them (not
their mean: each item's loss is divided by its whole minibatch's units). It takes an idle pass (`idle`: one that
learns nothing, under the reference with `reference`, with a backward pass with `gradient`) where a process has
fewer passes than the others, since the processes gather a sharded model's layers together; and clips its
gradient by the norm over every process's shard (`clip_gradients`, which returns the norm before). It may reduce a
minibatch's gradient once, in its last pass (`gradient_sync`, told before each of a minibatch's gradient passes
whether it is the last: a sharded adapter keeps the others' gradients in each process; none: each pass reduces its
own). After a pass that ran out of memory part way, it drops what that pass left (`recover`: a sharded model's
gradients not yet reduced, and its state of the pass), so that the next pass starts clean; a model sharded on one
process has it too, where the step goes on without the pass.

| Field | Type | Default | Description |
|---|---|---|---|
| `gradient_sync` | `Callable[[bool], None] \| None` | required |  |

**Methods**

- `def idle(self, *, gradient: bool = False, reference: bool = False) -> None`
- `def clip_gradients(self, maximum: float, ranks: Ranks) -> float`
- `def recover(self) -> None`

### `TrainablePolicy`

*class* · `implementations/rollout-objectives/src/rollout_objectives/step.py`

```python
class TrainablePolicy(Protocol)
```

What the step needs of a policy (`rollout_lora.policy.Policy` is one). An objective with a KL to the reference
or a preference loss against it needs `reference` too, one with an entropy bonus `logprobs_and_entropy`, and a
distillation over the teacher's top-k tokens `logprobs_among` (the logprobs of given tokens at each position,
beside the sampled ones'). A policy that runs packs is a `PackingPolicy`; a step shared among processes takes a
`SharedPolicy`.

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required |  |

**Methods**

- `def parameters(self) -> list[nn.Parameter]`
- `def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor`

## `rollout_objectives.packing`

Many segments in one row of a model's input, a prefix several share once.

### `Group`

*class* · `implementations/rollout-objectives/src/rollout_objectives/packing.py`

```python
class Group
```

Segments (by index) under a prefix of `shared` tokens they all start with (0 for a segment alone), and the
tokens the group puts in a row: its prefix once, then each segment's rest.

| Field | Type | Default | Description |
|---|---|---|---|
| `members` | `list[int]` | required |  |
| `shared` | `int` | required |  |
| `tokens` | `int` | required |  |

### `grouped`

*function* · `implementations/rollout-objectives/src/rollout_objectives/packing.py`

```python
def grouped(segments: Sequence[Segment], capacity: int, *, share: bool = True, least: int = SHARED_PREFIX) -> list[Group]
```

The segments in groups, each of which a pack holds whole. Sharing (`share`): the segments sorted by their
tokens, and each joins the group before it, under the prefix they all share, where that prefix is at least `least`
tokens, the group takes at most `capacity` tokens, and the group with it puts fewer tokens in a row than the group
and the segment apart (a segment that would cut a long prefix short starts a group of its own); else each segment
alone.

### `Pack`

*class* · `implementations/rollout-objectives/src/rollout_objectives/packing.py`

```python
class Pack
```

Some of the segments a step was given (`members`: their indices in what `packs` was given), in one row.

| Field | Type | Default | Description |
|---|---|---|---|
| `members` | `list[int]` | required |  |
| `segments` | `list[Segment]` | required |  |
| `tokens` | `list[int]` | `field(default_factory=list[int])` | The row. |
| `positions` | `list[int]` | `field(default_factory=list[int])` | Each row token's position in its segment. |
| `runs` | `list[Run]` | `field(default_factory=list[Run])` | The runs that make up the row, in its order: every root, then every branch. |
| `places` | `list[list[int]]` | `field(default_factory=list[list[int]])` | For each segment, where in the row each of its tokens is. |

**Methods**

- `@property def length(self) -> int` — Tokens in the row.
- `@property def segment_tokens(self) -> int` — Tokens of its segments, each counted in full (more than `length` where they share prefixes).
- `def scored(self, index: int, positions: Sequence[int] | None = None) -> tuple[list[int], list[int]]` — For the `index`-th segment: the row of the hidden state before each of `positions` (the sampled tokens'
  by default), and the token at each.
- `@classmethod def single(cls, member: int, segment: Segment) -> 'Pack'` — One segment alone.
- `@classmethod def laid_out(cls, segments: Sequence[Segment], groups: Sequence[Group], numbers: Sequence[int] | None = None) -> 'Pack'` — `groups` of `segments` in a row: each group's root (its prefix, or its one segment), then each shared
  prefix's branches. Its members are the segments' indices in `segments`, or their `numbers`.

### `packed` {#rollout_objectivespackingpacked}

*function* · `implementations/rollout-objectives/src/rollout_objectives/packing.py`

```python
def packed(segments: Sequence[Segment], groups: Sequence[Group], capacity: int) -> list[Pack]
```

`groups` of `segments` in packs of at most `capacity` tokens (a group longer than that in a pack of its own),
placed first-fit-decreasing by their tokens.

### `packs`

*function* · `implementations/rollout-objectives/src/rollout_objectives/packing.py`

```python
def packs(segments: Sequence[Segment], capacity: int, *, share: bool = True, least: int = SHARED_PREFIX) -> list[Pack]
```

`segments` in packs of at most `capacity` tokens, sharing prefixes of at least `least` tokens if `share`
(`grouped`, then `packed`).

### `Run` {#rollout_objectivespackingrun}

*class* · `implementations/rollout-objectives/src/rollout_objectives/packing.py`

```python
class Run
```

Tokens that are consecutive in a pack's row, and in their segment.

| Field | Type | Default | Description |
|---|---|---|---|
| `start` | `int` | required | Where in the row it starts. |
| `length` | `int` | required |  |
| `parent` | `int \| None` | `None` | For a branch, the index (in `Pack.runs`) of the root whose tokens come before its own; none for a root. |

**Methods**

- `@property def end(self) -> int`

### `sampled_positions`

*function* · `implementations/rollout-objectives/src/rollout_objectives/packing.py`

```python
def sampled_positions(segment: Segment) -> list[int]
```

The positions of the tokens the policy sampled in a segment.

### `Scores` {#rollout_objectivespackingscores}

*class* · `implementations/rollout-objectives/src/rollout_objectives/packing.py`

```python
class Scores
```

What a policy gives of one segment of a pack.

| Field | Type | Default | Description |
|---|---|---|---|
| `logprobs` | `torch.Tensor` | required | Of its sampled tokens. |
| `entropy` | `torch.Tensor \| None` | `None` | Of the policy's distribution at each sampled position, where asked for. |
| `among` | `torch.Tensor \| None` | `None` | The logprobs of given tokens at each sampled position (a row of them for each), where asked for. |

### `SHARED_PREFIX`

*constant* · `implementations/rollout-objectives/src/rollout_objectives/packing.py`

```python
SHARED_PREFIX = 32
```

The fewest tokens a prefix holds for segments to share it in a pack.

## `rollout_objectives.ranks`

The processes a step is shared among, one per GPU, and how a minibatch is shared.

### `Ranks`

*class* · `implementations/rollout-objectives/src/rollout_objectives/ranks.py`

```python
class Ranks
```

This process's place among those a step is shared among: its `rank` of `size`. `group` is the process group
their sums and gathers go through: one on the CPU (gloo), whatever the GPUs' own collectives run on (none: the
default group).

| Field | Type | Default | Description |
|---|---|---|---|
| `rank` | `int` | `0` |  |
| `size` | `int` | `1` |  |
| `group` | `Any` | `None` |  |

**Methods**

- `@property def shared(self) -> bool`
- `def summed(self, values: Sequence[float]) -> list[float]` — Each of `values` added up across the processes (in float64).
- `def gathered[T](self, value: T) -> list[T]` — Every process's `value` (picklable), by rank.
- `def most(self, value: float) -> float` — The largest of every process's `value`.

### `shares`

*function* · `implementations/rollout-objectives/src/rollout_objectives/ranks.py`

```python
def shares(sizes: Sequence[int], count: int) -> list[list[int]]
```

The indices of `sizes` (each a pass's tokens: a pack's, or a segment's) shared among `count` processes,
balanced by passes, then by tokens: each process takes at most its even share of the count (rounded up, as the
processes take their passes together), the largest first, each to the process with the fewest tokens so far of
those with room (the lower rank on a tie). Each process's indices are in their order in `sizes`. Every process
computes the same shares from the same sizes.

## `rollout_qwen`

Renderers for the Qwen model families.

### `qwen3`

*function* · `implementations/rollout-qwen/src/rollout_qwen/__init__.py`

```python
def qwen3(model: str | Tokenizer) -> Renderer
```

Qwen3: JSON tool calls, and thinking the model opens. `model` is a checkpoint's name, or its tokenizer.
A turn ends with `<|im_end|>`, or with the end of text, where the model stops too.

### `qwen35`

*function* · `implementations/rollout-qwen/src/rollout_qwen/__init__.py`

```python
def qwen35(model: str | Tokenizer) -> Renderer
```

Qwen3.5: XML function calls, and thinking the prompt opens. `model` is a checkpoint's name, or its tokenizer.
A turn ends with `<|im_end|>`, or with the end of text, where the model stops too.

## `rollout_gemma`

Renderers for the Gemma model families.

### `arguments`

*function* · `implementations/rollout-gemma/src/rollout_gemma/__init__.py`

```python
def arguments(text: str) -> dict[str, JsonValue]
```

A call's arguments (what is between its braces) as values: strings quoted with `<|"|>`, numbers, `true`,
`false`, `null`, objects in braces and lists in brackets; keys are bare (or quoted).

### `gemma4`

*function* · `implementations/rollout-gemma/src/rollout_gemma/__init__.py`

```python
def gemma4(model: str | Tokenizer) -> Renderer
```

Gemma 4, thinking: the template is asked for thinking, and the generation prompt opens the thought channel
(as Gemma's template itself does after a tool's response), so that a thinking budget can close it. `model` is a
checkpoint's name, or its tokenizer.

### `GemmaFunctionCalls`

*class* · `implementations/rollout-gemma/src/rollout_gemma/__init__.py`

```python
class GemmaFunctionCalls
```

`<|tool_call>call:name{key:<|"|>text<|"|>,count:3,flag:true,nested:{...},items:[...]}<tool_call|>`.

**Methods**

- `def parse(self, text: str, tools: Sequence[ToolSpecification]) -> tuple[str, list[ToolCall]]`

## `rollout_openai`

A model endpoint for the OpenAI Responses API, on an API key or a Codex login.

### `ApiKey`

*class* · `implementations/rollout-openai/src/rollout_openai/responses.py`

```python
class ApiKey
```

An OpenAI API key against the public Responses API.

| Field | Type | Default | Description |
|---|---|---|---|
| `key` | `str` | required |  |
| `base_url` | `str` | `'https://api.openai.com/v1'` |  |
| `accepts_max_output_tokens` | `bool` | `True` |  |

**Methods**

- `async def headers(self, client: httpx.AsyncClient, *, force_refresh: bool = False) -> dict[str, str]`
- `@property def url(self) -> str`

### `codex_provider`

*function* · `implementations/rollout-openai/src/rollout_openai/responses.py`

```python
def codex_provider(contract: ResponsesContract | None = None, *, blobs: Blobs | None = None) -> Callable[[DirectModel], ResponsesEndpoint]
```

An endpoint factory for `LocalRunner(providers={"codex": codex_provider()})`, using the local Codex login.

### `CodexLogin`

*class* · `implementations/rollout-openai/src/rollout_openai/responses.py`

```python
class CodexLogin
```

ChatGPT account tokens from a local Codex login.

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | `Path` | `Path.home() / '.codex' / 'auth.json'` |  |
| `url` | `str` | `'https://chatgpt.com/backend-api/codex/responses'` |  |
| `refresh_margin_seconds` | `int` | `300` |  |
| `accepts_max_output_tokens` | `bool` | `False` |  |

**Methods**

- `async def headers(self, client: httpx.AsyncClient, *, force_refresh: bool = False) -> dict[str, str]`

### `Credentials`

*class* · `implementations/rollout-openai/src/rollout_openai/responses.py`

```python
class Credentials(Protocol)
```

Where requests go and how they authenticate.

**Methods**

- `async def headers(self, client: httpx.AsyncClient, *, force_refresh: bool = False) -> dict[str, str]`
- `@property def url(self) -> str`
- `@property def accepts_max_output_tokens(self) -> bool` — Whether the backend accepts `max_output_tokens`.

### `hosted` {#rollout_openaihosted}

*function* · `implementations/rollout-openai/src/rollout_openai/responses.py`

```python
def hosted(model: str, *, api_key: str | None, context_limit: int, max_output_tokens: int, options: Mapping[str, JsonValue] | None = None, base_url: str | None = None) -> ResponsesEndpoint
```

The endpoint of a cluster's `api` provider for one of its models: its API key, the model's context and most
output (its catalog entry), the model's `options` (`reasoning_effort`: the effort a request is sampled with where
its binding says none), and the API's base URL (none: OpenAI's own). Raises `PermissionError` without a key.

### `ResponsesContract`

*class* · `implementations/rollout-openai/src/rollout_openai/responses.py`

```python
class ResponsesContract
```

The capability contract the endpoint advertises; the provider does not report it.

| Field | Type | Default | Description |
|---|---|---|---|
| `context_limit` | `int` | `200000` |  |
| `max_output_tokens` | `int` | `32000` |  |

### `ResponsesEndpoint`

*class* · `implementations/rollout-openai/src/rollout_openai/responses.py`

```python
class ResponsesEndpoint
```

Serves one model through the Responses API. Direct adapters do not deduplicate: a retried effect re-samples.

**Methods**

- `def __init__(self, credentials: Credentials, model: str, *, sampling: SamplingParameters | None = None, contract: ResponsesContract | None = None, client: httpx.AsyncClient | None = None, timeout: float = 600.0, blobs: Blobs | None = None) -> None` — `blobs` reads the bytes of `Media` blocks (images); without it, a context with media cannot be sent.
- `def describe(self, session_id: str) -> CapabilityContract`
- `async def cancel(self, effect_id: str) -> None` — Nothing to do: the request stops when the task awaiting `sample` is cancelled.
- `async def sample(self, request: SampleRequest, *, sampling: SamplingParameters | None = None) -> SampleResult` — One reply; `sampling` in place of the endpoint's own sampling parameters, for this request.
- `def request_body(self, request: SampleRequest, media: Mapping[str, bytes] | None = None, *, sampling: SamplingParameters | None = None) -> dict[str, JsonValue]` — The Responses API request for a sample request (public for tests and debugging). `media` holds the bytes
  of the context's `Media` blocks by SHA-256; `sampling`, the sampling parameters in place of the endpoint's.

## `rollout_anthropic`

A model endpoint for Anthropic's Messages API.

### `hosted` {#rollout_anthropichosted}

*function* · `implementations/rollout-anthropic/src/rollout_anthropic/messages.py`

```python
def hosted(model: str, *, api_key: str | None, context_limit: int, max_output_tokens: int, options: Mapping[str, JsonValue] | None = None, base_url: str | None = None, blobs: Blobs | None = None, timeout: float = 600.0) -> MessagesEndpoint
```

The endpoint of a cluster's `api` provider for one of its models (`endpoint = "rollout_anthropic:hosted"`): its
API key, the model's context and most output and its `options` (`MessagesOptions`), from its catalog entry, and the
API's base URL (none: Anthropic's own). Raises `PermissionError` without a key.

### `MessagesEndpoint`

*class* · `implementations/rollout-anthropic/src/rollout_anthropic/messages.py`

```python
class MessagesEndpoint
```

Serves one model through the Messages API. It does not deduplicate: a retried effect samples again.

**Methods**

- `def __init__(self, client: anthropic.AsyncAnthropic, model: str, *, contract: CapabilityContract, options: MessagesOptions | None = None, sampling: SamplingParameters | None = None, blobs: Blobs | None = None) -> None` — `client` is made with `max_retries=0` (the holder retries); `blobs` reads the bytes of `Media` blocks:
  without it, a context with media cannot be sent.
- `def describe(self, session_id: str) -> CapabilityContract`
- `async def cancel(self, effect_id: str) -> None` — Nothing to do: the request stops when the task awaiting `sample` is cancelled.
- `async def sample(self, request: SampleRequest, *, sampling: SamplingParameters | None = None) -> SampleResult` — One reply; `sampling` in place of the endpoint's own sampling parameters, for this request.
- `def request_body(self, request: SampleRequest, media: Mapping[str, bytes] | None = None, *, sampling: SamplingParameters | None = None) -> dict[str, Any]` — The Messages API request for a sample request (public for tests and debugging). `media` holds the bytes of
  the context's `Media` blocks by SHA-256; `sampling`, the sampling parameters in place of the endpoint's.

### `MessagesOptions`

*class* · `implementations/rollout-anthropic/src/rollout_anthropic/messages.py`

```python
class MessagesOptions
```

What one model takes, as its catalog entry's `options` say.

| Field | Type | Default | Description |
|---|---|---|---|
| `thinking` | `Literal['adaptive', 'budget', 'none']` | `'none'` |  |
| `effort` | `str \| None` | `None` | The effort adaptive thinking is steered by where a binding says none (none: the model's default). |
| `sampling` | `bool` | `True` | Whether it takes temperature and top-p. |
| `forced_tool_choice` | `bool` | `True` | Whether it takes a tool choice that forces a call (`any`, a named tool); where it does not, such a choice is sent as `auto`. |

**Methods**

- `@classmethod def of(cls, options: Mapping[str, JsonValue] | None) -> 'MessagesOptions'` — The options of a model's catalog entry (those not about the Messages API are left).

## `rollout_s3`

Blobs in S3 or any S3-compatible object store.

### `S3BlobStore`

*class* · `implementations/rollout-s3/src/rollout_s3/store.py`

```python
class S3BlobStore
```

Implements `Blobs` in an S3 bucket.

**Methods**

- `def __init__(self, bucket: str, *, prefix: str = 'blobs/', endpoint_url: str | None = None, region: str | None = None, client: 'S3Client | None' = None, refresh_after: float = REFRESH_AFTER, access_key_id_env: str | None = None, secret_access_key_env: str | None = None) -> None` — `client` replaces the boto3 client this store would create (e.g. with custom credentials).
  `access_key_id_env` and `secret_access_key_env` name the environment variables the store's key is read from
  (both, or neither: boto3's usual sources). Raises `ValueError` where one is named and not set.
- `@classmethod def from_url(cls, url: str, **options: Any) -> 'S3BlobStore'` — A store for `s3://bucket/prefix`.
- `def holds(self, reference: BlobReference) -> bool` — Whether a reference names an object of this store's bucket and prefix (`s3://BUCKET/PREFIX…`).
- `async def put(self, data: bytes, media_type: str) -> BlobReference`
- `async def read(self, reference: BlobReference) -> bytes`
- `async def delete(self, reference: BlobReference, *, unused_for: float = 0.0) -> None`
- `async def with_extension(self, reference: BlobReference, extension: str) -> str` — The URI of a copy of a blob's object named by its key and `extension` (`s3://BUCKET/KEY.zip`), made inside
  the bucket the first time it is asked for: for readers that tell an archive by its name, as Ray does a runtime
  environment's `working_dir`. Raises `FileNotFoundError` where the store does not have the blob.

## `rollout_runpod`

GPU pods on RunPod, and certificates for them from step-ca.

### `decrypted_key`

*function* · `implementations/rollout-runpod/src/rollout_runpod/certificates.py`

```python
def decrypted_key(encrypted: str, password: str) -> dict[str, Any]
```

A JWK provisioner's private key from its `encryptedKey` (a JWE, compact, encrypted with a password: PBES2 key
wrapping and AES-GCM content, as step makes it) and the password. Raises `ValueError` for a wrong password or a
JWE it does not read.

### `fingerprint`

*function* · `implementations/rollout-runpod/src/rollout_runpod/certificates.py`

```python
def fingerprint(root: bytes) -> str
```

The SHA-256 fingerprint of a certificate (PEM), in lowercase hexadecimal: what step-ca's clients trust a root
by.

### `Pod` {#rollout_runpodpod}

*class* · `implementations/rollout-runpod/src/rollout_runpod/api.py`

```python
class Pod
```

A pod, as RunPod says it is.

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required |  |
| `name` | `str` | required |  |
| `status` | `str` | required | RunPod's `desiredStatus`: `RUNNING`, `EXITED` (stopped) or `TERMINATED`. |
| `image` | `str` | `''` |  |
| `public_ip` | `str \| None` | `None` |  |
| `ports` | `Mapping[int, int]` | `field(default_factory=dict[int, int])` | Each exposed port of the pod, and the public port it is reached at. |
| `cost_per_hour` | `float \| None` | `None` |  |
| `gpu` | `str \| None` | `None` | The GPU type RunPod gave it, by its id (`NVIDIA H100 80GB HBM3`), where it says. |

**Methods**

- `def address(self, port: int = 8443) -> str | None` — Where `port` is reached from outside, `https://IP:PORT`; None until RunPod has said.
- `@classmethod def of(cls, said: Mapping[str, Any]) -> 'Pod'`

### `PodSpec`

*class* · `implementations/rollout-runpod/src/rollout_runpod/api.py`

```python
class PodSpec
```

A pod, as it is asked for.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | The pod's name, which its certificate's identity is made from (`spiffe://rollout/pod/NAME`). |
| `image` | `str` | required |  |
| `gpu_types` | `Sequence[str]` | required | RunPod's GPU type ids, in order of preference (`NVIDIA GeForce RTX 4090`, `NVIDIA H100 80GB HBM3`). |
| `gpu_count` | `int` | `1` |  |
| `env` | `Mapping[str, str]` | `field(default_factory=dict[str, str])` |  |
| `secrets` | `Mapping[str, str]` | `field(default_factory=dict[str, str])` | Variables whose values are secrets kept in RunPod's console, by the secret's name. |
| `sensitive` | `Mapping[str, str]` | `field(default_factory=dict[str, str], repr=False)` | Variables whose values are sent and never logged. |
| `ports` | `Sequence[str]` | `('8443/tcp',)` | `PORT/tcp` (a public port mapped to it; RunPod's HTTPS proxy would end TLS, so none is `/http`). |
| `volume_gb` | `int` | `50` |  |
| `volume_mount` | `str` | `'/workspace'` |  |
| `container_disk_gb` | `int` | `50` |  |
| `cloud` | `str` | `'SECURE'` | `SECURE` or `COMMUNITY`. |
| `data_centers` | `Sequence[str]` | `()` |  |
| `cuda_versions` | `Sequence[str]` | `()` | The CUDA versions the pod's machine may support (none: any); its driver must run the image's CUDA. |
| `interruptible` | `bool` | `False` |  |

**Methods**

- `def body(self) -> dict[str, Any]` — The pod as RunPod's API takes it.

### `RunPod`

*class* · `implementations/rollout-runpod/src/rollout_runpod/api.py`

```python
class RunPod
```

RunPod's pods API at `url`, with the key in the environment variable `key_env`.

**Methods**

- `def __init__(self, *, key_env: str = KEY, url: str = API, client: httpx.AsyncClient | None = None) -> None`
- `async def create(self, spec: PodSpec) -> Pod` — Ask for a pod; it starts as soon as RunPod has a machine for it.
- `async def pods(self, *, name: str | None = None) -> list[Pod]` — Every pod of the account (those named `name`, if given).
- `async def pod(self, id: str) -> Pod`
- `async def start(self, id: str) -> None` — Start a stopped pod (its volume as it was left; its container disk afresh).
- `async def stop(self, id: str) -> None` — Stop a pod: its GPU is released and no longer billed; its volume is kept (and billed) until it is deleted.
- `async def terminate(self, id: str) -> None` — Delete a pod and its volume.
- `async def aclose(self) -> None`

### `RunPodError`

*class* · `implementations/rollout-runpod/src/rollout_runpod/api.py`

```python
class RunPodError(Exception)
```

RunPod refused a request, or could not be reached. Says the request, the status and RunPod's message: never
the key.

**Methods**

- `def __init__(self, message: str, status: int | None = None) -> None`

### `StepCa`

*class* · `implementations/rollout-runpod/src/rollout_runpod/certificates.py`

```python
class StepCa
```

step-ca at `url`, whose root certificate is `root` (PEM), with the JWK provisioner `provisioner` whose private
key is `key` (a JWK, EC P-256). Its own TLS is checked by the root; with `system`, by the system's roots (behind a
proxy that ends TLS with a public certificate, such as a Cloudflare Tunnel).

**Methods**

- `def __init__(self, url: str, *, provisioner: str, key: Mapping[str, Any], root: bytes, client: httpx.AsyncClient | None = None, system: bool = False) -> None`
- `@classmethod def from_files(cls, url: str, *, provisioner: str, key: Path, root: Path, system: bool = False) -> 'StepCa'` — With the provisioner's key and the root certificate read from files.
- `def token(self, subject: str, sans: Sequence[str] | None = None, *, audience: str = 'sign', lifetime: float = TOKEN_LIFETIME, now: float | None = None, pinned: bool = True) -> str` — A one-time token for a certificate of `subject` (with `sans`, by default the subject alone), good for
  `lifetime` seconds; with `audience = "revoke"`, for revoking the certificate whose serial `subject` is.
  `pinned` names the root's fingerprint in it (`sha`): step then trusts only that root for step-ca's own TLS,
  whatever `--root` says, which fails behind a proxy that ends TLS with a public certificate.
- `def pod_token(self, identity: str, *, lifetime: float = TOKEN_LIFETIME) -> str` — The one-time token a pod gets its first certificate with: its identity as the subject and the only SAN. It
  names no root (`pinned` false): the pod is given the cluster's root and checks step-ca's TLS by its `--root`.
- `async def certificate(self, identity: str, *, lifetime: float | None = None) -> tuple[bytes, bytes]` — A certificate for `identity` (its subject and its only URI SAN), from a key made here: the certificate with
  its chain, and the key (both PEM). `lifetime` asks for fewer seconds than the provisioner's default.
- `async def revoke(self, serial: str, *, reason: str = '') -> None` — Revoke the certificate whose serial is `serial` (decimal), so that it is not renewed (passive revocation).
- `async def aclose(self) -> None`

## `rollout_tinker`

A trainer and an engine at Thinking Machines (Tinker).

### `TinkerEngine`

*class* · `implementations/rollout-tinker/src/rollout_tinker/engine.py`

```python
class TinkerEngine
```

Samples `model` (Tinker's id for it), or the sampler checkpoint an adapter's pointer names. `max_model_len` is
the longest turn it takes (prompt and completion; Tinker's context for `Qwen/Qwen3.5-9B` is 64K). `service` is
as `TinkerTrainer` takes it.

| Field | Type | Default | Description |
|---|---|---|---|
| `processes` | `Sequence[int]` | `()` |  |

**Methods**

- `def __init__(self, model: str, *, max_model_len: int = 32768, project: str | None = None, service: 'Service | str | None' = None) -> None`
- `async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float, stop_token_ids: Sequence[int], adapter: str | None, top: int = 0) -> Generation`
- `async def score(self, tokens: Sequence[int], *, start: int, end: int | None = None, top: int = 0, adapter: str | None) -> Scores`
- `async def load_adapter(self, name: str, path: str) -> None` — Sample from the sampler checkpoint that the pointer in `path` (a version's weights) names.
- `async def remove_adapter(self, name: str) -> None`
- `async def load_weights(self, path: str) -> None`
- `async def sleep(self) -> None` — Nothing to free on this machine.
- `async def wake(self) -> None` — Nothing was freed.
- `def close(self) -> None`

### `TinkerSettings`

*class* · `implementations/rollout-tinker/src/rollout_tinker/settings.py`

```python
class TinkerSettings(StepSettings)
```

Tinker scales an adapter by its own `lora_alpha / rank`, not by our twice the rank.

| Field | Type | Default | Description |
|---|---|---|---|
| `learning_rate` | `float` | `0.0001` | Twice `LoraTrainer`'s default: Tinker's adapters have an alpha of 32 (its archives say so), half our scale at rank 32, and Adam moves a weight by about the rate whatever its scale. |
| `tokens_per_step` | `int` | `65536` | Sampled tokens per optimizer step. A step that is one optimizer step needs no pass for the logprobs it starts from; every optimizer step costs Tinker at least one of its clock cycles. |
| `project` | `str \| None` | `None` | A Tinker project's id (not a secret); else `TINKER_PROJECT_ID`, if set. |

### `TinkerTrainer`

*class* · `implementations/rollout-tinker/src/rollout_tinker/trainer.py`

```python
class TinkerTrainer
```

Trains a LoRA adapter over `model` at Thinking Machines, one step at a time: a step starts from the training
state its parent names (its optimizer too, if it is given the parent's state) and leaves pointers to the new
checkpoints (which Tinker's bridge, `rollout_tinker.bridges`, turns into an adapter engines here load). A client
from the step before is used again when the parent is the state it saved. `service` is what calls Tinker: by
default a session the SDK opens with the key it finds; `module:name` of what makes another (a test names a fake
one so). `settings` are `TinkerSettings`' (its `objective` among them); those in `CHANGEABLE`, and the changeable
components of its objective, it takes between steps (`rollout_train.trainer.Changeable`). Raises `ValueError` for
an objective that reads the reference, the entropy or the top-k form's logprobs, which Tinker does not give here.

| Field | Type | Default | Description |
|---|---|---|---|
| `weights` |  | `'lora'` |  |

**Methods**

- `def __init__(self, model: str, *, service: 'Service | str | None' = None, **settings: Any) -> None`
- `@property def objective(self) -> Objective`
- `@property def changeable(self) -> Mapping[str, JsonValue]`
- `def change(self, settings: Mapping[str, JsonValue]) -> None`
- `async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step`

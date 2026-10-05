{{- define "rollout.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag }}
{{- end }}

{{- define "rollout.labels" -}}
app.kubernetes.io/part-of: rollout
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "rollout.ledgerUrl" -}}
postgresql://rollout@postgres.{{ .Release.Namespace }}:5432/rollout
{{- end }}

{{/* What each role is given is what its code reads, and nothing more (docs/deploy/helm.md#what-each-role-is-given):
each of these is a part of a container's environment, and `rollout.volumes` and `rollout.volumeMounts` take the names
of the volumes a role mounts. A run's job is given everything a run uses (`rollout.runEnv`). */}}

{{/* The cluster config, which every process of the platform reads. */}}
{{- define "rollout.configEnv" -}}
- name: ROLLOUT_CLUSTER
  value: /etc/rollout/cluster.toml
{{- end }}

{{/* The ledger's password: its URL (in the cluster config) holds none. */}}
{{- define "rollout.ledgerEnv" -}}
- name: PGPASSWORD
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.stores }}, key: POSTGRES_PASSWORD}}
{{- end }}

{{/* The blob store's endpoint and keys (the store's root keys: versitygw has no others), for what reads or writes
blobs. */}}
{{- define "rollout.blobsEnv" -}}
- name: AWS_ENDPOINT_URL
  value: http://s3.{{ .Release.Namespace }}:7070
- name: AWS_DEFAULT_REGION
  value: us-east-1
- name: AWS_ACCESS_KEY_ID
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.stores }}, key: ROOT_ACCESS_KEY_ID}}
- name: AWS_SECRET_ACCESS_KEY
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.stores }}, key: ROOT_SECRET_ACCESS_KEY}}
{{- end }}

{{/* A second blob store's keys ([stores.r2]), each optional: those `keys` names (WRITER_…, READER_…), from `root`. */}}
{{- define "rollout.r2Env" -}}
{{- range $key := .keys }}
- name: R2_{{ $key }}
  valueFrom: {secretKeyRef: {name: {{ $.root.Values.secrets.r2 }}, key: {{ $key }}, optional: true}}
{{- end }}
{{- end }}

{{/* The platform's token for the ledger service: what the service checks tokens against, and what runs' drivers sign
pods' tokens with. */}}
{{- define "rollout.ledgerTokenEnv" -}}
- name: ROLLOUT_LEDGER_TOKEN
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.ledger }}, key: ROLLOUT_LEDGER_TOKEN, optional: true}}
{{- end }}

{{/* Tinker's key, for what trains or samples on Tinker (a run's job). */}}
{{- define "rollout.tinkerEnv" -}}
- name: TINKER_API_KEY
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.tinker }}, key: TINKER_API_KEY, optional: true}}
{{- end }}

{{/* Where models' tokenizers and weights are kept, on the state volume. */}}
{{- define "rollout.cacheEnv" -}}
- name: HF_HOME
  value: {{ .Values.state.path }}/huggingface
{{- end }}

{{/* What reaches RunPod: its API key (runs' drivers, which lease pods, and the reaper). */}}
{{- define "rollout.podEnv" -}}
- name: RUNPOD_API_KEY
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.runpod }}, key: RUNPOD_API_KEY, optional: true}}
{{- end }}

{{/* The hosted APIs' keys, for what samples them (the gateway, each run's job): from the Secret `secrets.providers`,
each key optional, so a provider whose key is missing is refused when it is asked, and the rest go on. */}}
{{- define "rollout.providerEnv" -}}
- name: OPENAI_API_KEY
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.providers }}, key: OPENAI_API_KEY, optional: true}}
- name: ANTHROPIC_API_KEY
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.providers }}, key: ANTHROPIC_API_KEY, optional: true}}
{{- end }}

{{/* Everything a run uses, for each run's job (files/rayjob.yaml): the stores, both stores' keys, Tinker's, the
ledger service's token, the hosted APIs' keys and RunPod's. */}}
{{- define "rollout.runEnv" -}}
{{ include "rollout.configEnv" . }}
{{ include "rollout.ledgerEnv" . }}
{{ include "rollout.blobsEnv" . }}
{{- include "rollout.r2Env" (dict "root" . "keys" (list "WRITER_ACCESS_KEY_ID" "WRITER_SECRET_ACCESS_KEY" "READER_ACCESS_KEY_ID" "READER_SECRET_ACCESS_KEY")) }}
{{ include "rollout.tinkerEnv" . }}
{{ include "rollout.ledgerTokenEnv" . }}
{{ include "rollout.providerEnv" . }}
{{ include "rollout.podEnv" . }}
{{ include "rollout.cacheEnv" . }}
{{- end }}

{{/* What a process outside the Ray cluster needs to reach it: the cluster's token (KubeRay keeps it in a Secret named
after the cluster, and gives it to Ray's own pods itself). */}}
{{- define "rollout.rayClientEnv" -}}
- name: RAY_AUTH_MODE
  value: token
- name: RAY_AUTH_TOKEN
  valueFrom: {secretKeyRef: {name: {{ .Values.ray.name }}, key: auth_token}}
{{- end }}

{{/* An egress rule to the cluster's DNS (`networkPolicies.dns`). */}}
{{- define "rollout.dnsEgress" -}}
{{- $dns := .Values.networkPolicies.dns }}
to:
  - namespaceSelector: {matchLabels: {kubernetes.io/metadata.name: {{ $dns.namespace }}}}
    podSelector:
      matchLabels: {{- toYaml $dns.podLabels | nindent 8 }}
ports: [{port: 53, protocol: UDP}, {port: 53, protocol: TCP}]
{{- end }}

{{/* An egress rule to every address but the private ranges (`networkPolicies.privateRanges`): the internet. */}}
{{- define "rollout.internetEgress" -}}
to:
  - ipBlock:
      cidr: 0.0.0.0/0
      except: {{- toYaml .Values.networkPolicies.privateRanges | nindent 8 }}
{{- end }}

{{/* The checks `templates/admission.yaml` makes of each pod spec of a RayJob, at the CEL path given: a YAML list of
`expression` and `message`, `@@` standing for the path. */}}
{{- define "rollout.podChecks" -}}
{{- $checks := list
  (dict "message" "it shares none of the node's network, processes or IPC, and has no ephemeral containers"
        "expression" "!(has(@@.hostNetwork) && @@.hostNetwork) && !(has(@@.hostPID) && @@.hostPID) && !(has(@@.hostIPC) && @@.hostIPC) && (!has(@@.ephemeralContainers) || size(@@.ephemeralContainers) == 0)")
  (dict "message" "it runs under the namespace's default account"
        "expression" "(!has(@@.serviceAccountName) || @@.serviceAccountName in ['', 'default']) && (!has(@@.serviceAccount) || @@.serviceAccount in ['', 'default'])")
  (dict "message" "its volumes are ConfigMaps, empty directories, the downward API, projections and the run's Secrets and claims"
        "expression" "!has(@@.volumes) || @@.volumes.all(v, !has(v.hostPath) && (has(v.configMap) || has(v.emptyDir) || has(v.downwardAPI) || has(v.projected) || (has(v.secret) && has(v.secret.secretName) && v.secret.secretName in variables.secrets) || (has(v.persistentVolumeClaim) && v.persistentVolumeClaim.claimName in variables.claims)) && (!has(v.projected) || !has(v.projected.sources) || v.projected.sources.all(s, !has(s.secret) || (has(s.secret.name) && s.secret.name in variables.secrets))))")
}}
{{- range $list, $said := dict "containers" "container" "initContainers" "init container" }}
{{- $checks = append $checks (dict "message" (printf "no %s is privileged, adds capabilities or takes a host port" $said)
      "expression" (printf "!has(@@.%s) || @@.%s.all(c, (!has(c.securityContext) || ((!has(c.securityContext.privileged) || !c.securityContext.privileged) && (!has(c.securityContext.capabilities) || !has(c.securityContext.capabilities.add) || size(c.securityContext.capabilities.add) == 0))) && (!has(c.ports) || c.ports.all(p, !has(p.hostPort) || p.hostPort == 0)))" $list $list)) }}
{{- $checks = append $checks (dict "message" (printf "no %s reads a Secret but the run's" $said)
      "expression" (printf "!has(@@.%s) || @@.%s.all(c, (!has(c.env) || c.env.all(e, !has(e.valueFrom) || !has(e.valueFrom.secretKeyRef) || (has(e.valueFrom.secretKeyRef.name) && e.valueFrom.secretKeyRef.name in variables.secrets))) && (!has(c.envFrom) || c.envFrom.all(e, !has(e.secretRef) || (has(e.secretRef.name) && e.secretRef.name in variables.secrets))))" $list $list)) }}
{{- end }}
{{- range $checks }}
- message: {{ .message | quote }}
  expression: {{ .expression | replace "@@" $ | quote }}
{{- end }}
{{- end }}

{{/* The pod that publishes the root, the provisioner's key and the gateway's certificate (templates/step-ca.yaml), as a
Job's or a CronJob's template: given Values, Release and step-ca's in-cluster url. */}}
{{- define "rollout.pkiPod" -}}
metadata: {labels: {app: pki}}
spec:
  serviceAccountName: pki
  restartPolicy: OnFailure
  containers:
    - name: publish
      image: {{ .Values.image.repository }}:{{ .Values.image.tag }}
      imagePullPolicy: {{ .Values.image.pullPolicy }}
      command: [rollout, pki, publish, --ca-directory, /home/step, --url, {{ .url | quote }}, --namespace,
                {{ .Release.Namespace | quote }}, --secret, {{ .Values.secrets.stepCa | quote }}, --tls-secret,
                {{ .Values.secrets.gatewayTls | quote }}, --provisioner, {{ .Values.stepCa.provisioner | quote }}]
      env:
        - name: STEP_CA_PASSWORD
          valueFrom: {secretKeyRef: {name: {{ .Values.stepCa.passwordSecret }}, key: password}}
      volumeMounts: [{name: ca, mountPath: /home/step, readOnly: true}]
      resources: {requests: {cpu: 50m, memory: 128Mi}, limits: {memory: 256Mi}}
  volumes:
    - name: ca
      persistentVolumeClaim: {claimName: data-step-ca-0, readOnly: true}
{{- end }}

{{/* Where each volume is mounted: the state volume, the ConfigMap `rollout`, and the Secrets that are files (the
gateway's keys, Tinker's credentials, step-ca's root and provisioner key, the gateway's certificate). */}}
{{- define "rollout.mountPaths" -}}
state: {{ .Values.state.path }}
config: /etc/rollout
gateway-keys: /etc/rollout-secrets/gateway
tinker: /root/.tinker
step-ca: /etc/rollout-secrets/step-ca
gateway-tls: /etc/rollout-secrets/tls
{{- end }}

{{/* The mounts of the volumes `names` names (of `rollout.mountPaths`), from `root`. */}}
{{- define "rollout.volumeMounts" -}}
{{- $paths := include "rollout.mountPaths" .root | fromYaml }}
{{- range $name := .names }}
- {name: {{ $name }}, mountPath: {{ index $paths $name }}{{ if ne $name "state" }}, readOnly: true{{ end }}}
{{- end }}
{{- end }}

{{/* The volumes `names` names, from `root`: each Secret optional, readable by its owner only. */}}
{{- define "rollout.volumes" -}}
{{- $root := .root }}
{{- $secrets := dict "gateway-keys" $root.Values.secrets.gatewayKeys "tinker" $root.Values.secrets.tinker "step-ca" $root.Values.secrets.stepCa "gateway-tls" $root.Values.secrets.gatewayTls }}
{{- range $name := .names }}
{{- if eq $name "state" }}
- name: state
  persistentVolumeClaim: {claimName: {{ $root.Values.state.claim }}}
{{- else if eq $name "config" }}
- name: config
  configMap:
    name: rollout
    items:
      - {key: cluster.toml, path: cluster.toml}
      - {key: rayjob.yaml, path: rayjob.yaml}
      {{- range $path, $_ := $root.Files.Glob "files/presets/*.toml" }}
      - {key: {{ trimPrefix "files/" $path | replace "/" "_" }}, path: {{ trimPrefix "files/" $path }}}
      {{- end }}
{{- else }}
- name: {{ $name }}
  secret: {secretName: {{ index $secrets $name }}, optional: true, defaultMode: 0400}
{{- end }}
{{- end }}
{{- end }}

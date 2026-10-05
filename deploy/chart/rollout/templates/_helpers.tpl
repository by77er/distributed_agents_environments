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

{{/* What every process of the platform is given: the cluster config, and the stores' endpoint and credentials. The
ledger's URL (in the cluster config) holds no password: PGPASSWORD does. */}}
{{- define "rollout.env" -}}
- name: ROLLOUT_CLUSTER
  value: /etc/rollout/cluster.toml
- name: PGPASSWORD
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.stores }}, key: POSTGRES_PASSWORD}}
- name: AWS_ENDPOINT_URL
  value: http://s3.{{ .Release.Namespace }}:7070
- name: AWS_DEFAULT_REGION
  value: us-east-1
- name: AWS_ACCESS_KEY_ID
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.stores }}, key: ROOT_ACCESS_KEY_ID}}
- name: AWS_SECRET_ACCESS_KEY
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.stores }}, key: ROOT_SECRET_ACCESS_KEY}}
- name: TINKER_API_KEY
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.tinker }}, key: TINKER_API_KEY, optional: true}}
- name: ROLLOUT_LEDGER_TOKEN
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.ledger }}, key: ROLLOUT_LEDGER_TOKEN, optional: true}}
{{- range $key := list "WRITER_ACCESS_KEY_ID" "WRITER_SECRET_ACCESS_KEY" "READER_ACCESS_KEY_ID" "READER_SECRET_ACCESS_KEY" }}
- name: R2_{{ $key }}
  valueFrom: {secretKeyRef: {name: {{ $.Values.secrets.r2 }}, key: {{ $key }}, optional: true}}
{{- end }}
- name: HF_HOME
  value: {{ .Values.state.path }}/huggingface
{{- end }}

{{/* The hosted APIs' keys, for what samples them (the gateway, each run's job): from the Secret `secrets.providers`,
each key optional, so a provider whose key is missing is refused when it is asked, and the rest go on. */}}
{{/* What reaches RunPod: its API key (runs' drivers, which lease pods, and the reaper). */}}
{{- define "rollout.podEnv" -}}
- name: RUNPOD_API_KEY
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.runpod }}, key: RUNPOD_API_KEY, optional: true}}
{{- end }}

{{- define "rollout.providerEnv" -}}
- name: OPENAI_API_KEY
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.providers }}, key: OPENAI_API_KEY, optional: true}}
- name: ANTHROPIC_API_KEY
  valueFrom: {secretKeyRef: {name: {{ .Values.secrets.providers }}, key: ANTHROPIC_API_KEY, optional: true}}
{{- end }}

{{/* What a process outside the Ray cluster needs to reach it: the cluster's token (KubeRay keeps it in a Secret named
after the cluster, and gives it to Ray's own pods itself). */}}
{{- define "rollout.rayClientEnv" -}}
- name: RAY_AUTH_MODE
  value: token
- name: RAY_AUTH_TOKEN
  valueFrom: {secretKeyRef: {name: {{ .Values.ray.name }}, key: auth_token}}
{{- end }}

{{- define "rollout.volumeMounts" -}}
- {name: state, mountPath: {{ .Values.state.path }}}
- {name: config, mountPath: /etc/rollout, readOnly: true}
- {name: gateway-keys, mountPath: /etc/rollout-secrets/gateway, readOnly: true}
- {name: tinker, mountPath: /root/.tinker, readOnly: true}
- {name: step-ca, mountPath: /etc/rollout-secrets/step-ca, readOnly: true}
- {name: gateway-tls, mountPath: /etc/rollout-secrets/tls, readOnly: true}
{{- end }}

{{- define "rollout.volumes" -}}
- name: state
  persistentVolumeClaim: {claimName: {{ .Values.state.claim }}}
- name: config
  configMap:
    name: rollout
    items:
      - {key: cluster.toml, path: cluster.toml}
      - {key: rayjob.yaml, path: rayjob.yaml}
      {{- range $path, $_ := .Files.Glob "files/presets/*.toml" }}
      - {key: {{ trimPrefix "files/" $path | replace "/" "_" }}, path: {{ trimPrefix "files/" $path }}}
      {{- end }}
- name: gateway-keys
  secret: {secretName: {{ .Values.secrets.gatewayKeys }}, optional: true, defaultMode: 0400}
- name: tinker
  secret: {secretName: {{ .Values.secrets.tinker }}, optional: true, defaultMode: 0400}
- name: step-ca
  secret: {secretName: {{ .Values.secrets.stepCa }}, optional: true, defaultMode: 0400}
- name: gateway-tls
  secret: {secretName: {{ .Values.secrets.gatewayTls }}, optional: true, defaultMode: 0400}
{{- end }}

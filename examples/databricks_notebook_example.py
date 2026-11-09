# Databricks notebook source
# MAGIC %md
# MAGIC # listen-in: one station per task
# MAGIC
# MAGIC Records one radio station, transcribes it locally with Whisper on the driver, and
# MAGIC (optionally) summarises the news in English with Claude. Outputs go to a Unity
# MAGIC Catalog Volume.
# MAGIC
# MAGIC **Compute:** classic or dedicated (single-user) cluster with the init script
# MAGIC `examples/databricks_init_script.sh` (installs ffmpeg). GPU compute is faster.
# MAGIC On serverless, install the `ffmpeg` extra instead (bundled static binary).
# MAGIC
# MAGIC **Scheduling:** create a Databricks Job with one notebook task per station and a
# MAGIC cron schedule such as `0 0,30 * * * ?` (minute 0 and 30). Do not loop or sleep in
# MAGIC this notebook; the Job is the scheduler.

# COMMAND ----------

# MAGIC %pip install "listen-in[whisper,llm-openai,llm-anthropic] @ git+https://github.com/HWDSIU/listen-in.git@v0.2.0"

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821 - provided by Databricks

# COMMAND ----------

FRENCH_STATION = "https://www.radio-browser.info/history/31074f8a-e6f4-11e9-a96c-52543be04c81"
ENGLISH_STATION = "https://www.radio-browser.info/history/598c4d0e-6b06-43fb-bff4-717c591213a9"

# Job parameters / widgets: one station per task so stations schedule independently.
dbutils.widgets.text("station", FRENCH_STATION, "Station (URL, UUID or history URL)")  # noqa: F821
dbutils.widgets.text("language", "fr", "Language hint (ISO 639-1, e.g. fr, en)")  # noqa: F821
dbutils.widgets.text("duration_seconds", "600", "Seconds to record")  # noqa: F821
dbutils.widgets.text("output_dir", "/Volumes/main/radio/listen_in/outputs", "Output Volume path")  # noqa: F821
# Whisper model: "small" for English, French and most languages; "turbo" for less widely
# spoken languages such as Swahili (use a GPU cluster or a driver with 16 GB+ RAM).
dbutils.widgets.text("whisper_model", "small", "Whisper model (small | turbo)")  # noqa: F821
dbutils.widgets.dropdown("llm_provider", "azure_openai", ["none", "azure_openai", "openai", "anthropic"],  # noqa: F821
                         "LLM provider (none = transcription only)")
dbutils.widgets.text("llm_model", "", "LLM model / Azure deployment name")  # noqa: F821
dbutils.widgets.text("llm_endpoint", "", "Azure endpoint (https://<resource>.openai.azure.com)")  # noqa: F821
dbutils.widgets.text("llm_api_version", "2024-12-01-preview", "Azure API version")  # noqa: F821

station = dbutils.widgets.get("station")  # noqa: F821
language = dbutils.widgets.get("language")  # noqa: F821
duration_seconds = int(dbutils.widgets.get("duration_seconds"))  # noqa: F821
output_dir = dbutils.widgets.get("output_dir")  # noqa: F821
whisper_model = dbutils.widgets.get("whisper_model")  # noqa: F821
llm_provider = dbutils.widgets.get("llm_provider")  # noqa: F821
llm_model = dbutils.widgets.get("llm_model").strip()  # noqa: F821
llm_endpoint = dbutils.widgets.get("llm_endpoint").strip()  # noqa: F821
llm_api_version = dbutils.widgets.get("llm_api_version").strip()  # noqa: F821

# COMMAND ----------

from listen_in import process_station


def get_llm_api_key() -> str:
    """Fetch the LLM key. Adapt to how your workspace stores secrets."""
    # Option A: a Databricks secret scope (can be backed by Azure Key Vault).
    #   databricks secrets put-secret listen-in llm-api-key
    return dbutils.secrets.get(scope="listen-in", key="llm-api-key")  # noqa: F821
    # Option B: Azure Key Vault directly, with the cluster's Azure identity:
    # from azure.identity import DefaultAzureCredential
    # from azure.keyvault.secrets import SecretClient
    # return SecretClient(vault_url="https://<vault>.vault.azure.net",
    #                     credential=DefaultAzureCredential()).get_secret("<secret-name>").value


llm = None
if llm_provider != "none":
    llm = {"provider": llm_provider, "model": llm_model or None, "api_key": get_llm_api_key()}
    if llm_provider == "azure_openai":
        llm.update(base_url=llm_endpoint, api_version=llm_api_version)

result = process_station(
    station=station,
    language=language,
    duration_seconds=duration_seconds,
    output_dir=output_dir,          # persistent UC Volume; audio is staged on local /tmp first
    whisper_model=whisper_model,
    llm=llm,
)

display(result.model_dump(mode="json"))  # noqa: F821 - the API key is never included

# COMMAND ----------

# Fail the Job task (so alerts fire) when capture or transcription failed.
# Analysis failures keep the transcript and are reported in the manifest.
if result.statuses["capture"] != "ok" or result.statuses["transcribe"] != "ok":
    raise RuntimeError(f"listen-in run {result.run_id} failed: {result.stages}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: load manifests into a Delta table
# MAGIC ```python
# MAGIC df = spark.read.option("multiLine", True).json(f"{output_dir}/manifests/")
# MAGIC df.write.mode("append").saveAsTable("main.radio.listen_in_runs")
# MAGIC ```

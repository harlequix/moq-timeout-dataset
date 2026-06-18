DATADIR  ?= data

SUMMARY    := $(DATADIR)/summary.csv
LATENCY    := $(DATADIR)/latency.csv
SUM_TABLE  := $(DATADIR)/summary_table.txt
DECODAB    := $(DATADIR)/decodability.csv
STALLS     := $(DATADIR)/stalls.csv
STALL_SUM  := $(DATADIR)/stall_summary.csv
ABANDON    := $(DATADIR)/abandonment.csv

.PHONY: all clean

all: $(SUMMARY) $(LATENCY) $(DECODAB) $(STALLS) $(STALL_SUM) $(ABANDON)

$(SUMMARY) $(LATENCY) $(SUM_TABLE): $(wildcard $(DATADIR)/*/*/run*/relay_metrics.csv)
	bash analysis/aggregate-results.sh $(DATADIR)

$(DECODAB): $(LATENCY)
	uv run python3 analysis/decodability.py $(LATENCY)

$(STALLS): $(wildcard $(DATADIR)/*/*/run*/sub1_display.csv)
	uv run python3 analysis/export-stalls.py $(DATADIR)

$(STALL_SUM): $(wildcard $(DATADIR)/*/*/run*/sub1_display.csv)
	uv run python3 analysis/summarize-stalls.py $(DATADIR)

$(ABANDON): $(wildcard $(DATADIR)/*/*/run*/relay.log)
	uv run python3 analysis/abandonment.py $(DATADIR)

clean:
	rm -f $(SUMMARY) $(LATENCY) $(SUM_TABLE) $(DECODAB) $(STALLS) $(STALL_SUM) $(ABANDON)

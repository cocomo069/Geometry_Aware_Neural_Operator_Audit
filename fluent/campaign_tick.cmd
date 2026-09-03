@echo off
REM FluentCampaign tick -- run one Fluent solve per invocation (Task Scheduler).
REM stdout goes to a SEPARATE file: campaign_tick.py opens fluent_campaign.log itself,
REM and Windows will not let two handles append to the same file concurrently.
cd /d "D:\Personal Projects\geom_aware_neural_operator"
".venv\Scripts\python.exe" fluent\campaign_tick.py >> logs\fluent_campaign.out.log 2>&1

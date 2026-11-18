# Japanese number and date development controls

This document describes the authoring-stage proposal for issue #120. The proposal is not adopted until independent review and merge are recorded in issue #120. The issue record will hold the adopted commit and bank digest.

The two source-near examples use the same exhibition event, active predicate, unspoken organizer, and Japanese numeral. One gives a calendar date without a clock. The other gives a duration of three days. The duration row does not say that the days have already elapsed. Their Japanese forms are related, but they are not a one-token minimal pair.

| ID | Japanese | Taiwan Traditional Chinese reference | Purpose |
| --- | --- | --- | --- |
| ND-01-CALENDAR-DATE | 展示会を三月三日に開催します。 | 三月三日舉辦展覽會。 | Calendar date, without clock time or duration. |
| ND-02-DURATION-DAYS | 展示会を三日間開催します。 | 舉辦為期三天的展覽會。 | Duration of three days, without calendar date or clock time. |

Existing DEV controls provide nearby but distinct material: N5X-05/N5X-06 cover つ and 本 counters; N5X-09/N5X-10 cover floor numbers; jz074/jz075/jz077/jz078 cover animal, 本, population, and pencil counts. jz081 has a full-year date; jz083 has a day within a relative month; jz080 and CB-B14/CB-B15 combine dates with clock readings. None is the same date-versus-duration-in-days contrast.

Both rows are development-visible first-party MIT proposals. Their references and variants are provisional. They were authored before any new-row engine execution. Independent review is pending. The rows do not establish broad N5 quantity or date coverage, and they do not show how the current runtime tokenizes these forms.

The complete API fixture is copied from the first context in the fixed 269-case DEV manifest. The new bank is not registered or executed. Do not measure a corresponding rule with these rows until adoption and merge are recorded in issue #120.

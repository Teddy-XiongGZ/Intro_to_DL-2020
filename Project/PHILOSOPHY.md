## Refining the scope -> for Language and Cognition
```text
Causal relations are context-dependent.

The following tuple is from the third entry of COPA. <a2> apparently is the more proper answer.

    <p>The women met for coffee.</p>
    <a1>The cafe reopened in a new location.</a1>
    <a2>They wanted to catch up with each other.</a2>

But if we know from the context that "there is a custom in this little town that when shops reopen in a new location, all townswomen conventionally gather for a party", <a1> will also seem plausible.

First entry of COPA:

    <p>My body cast a shadow over the grass.</p>
    <a1>The sun was rising.</a1>
    <a2>The grass was cut.</a2>

If we know from the context that it was midnight and it was actually not the sun but the streetlights that are causing <p>, is <a1> still the prefered choice? We are not forced, though, to pick an answer between <a1> and <a2>; the output of our model can be a score that indicates how closely the two events are causally related.

The problem with this is that the context often originates from common sense. How are we to include common sense in our model?
```
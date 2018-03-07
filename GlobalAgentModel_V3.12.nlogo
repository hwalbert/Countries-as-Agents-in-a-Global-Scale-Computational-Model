extensions [nw]

globals[
  global-wealth
  num-active-countries-each-year ;;in the tribute model described by Axelrod 3 out of 10 agents were active each year. in this model 30% of the total number of agents are asked to be active each year
  active-country ;;One specific country that is active at any given time
  current-target ;;the country selected for targeting...this target is chosen if they have the highest attack score of all the potential targets
  last-year-wealth ;;last years wealth for updating plots
  global-wealth-change
  number-of-wars
  num-tributes
  war-to-tribute-ratio
  sum-of-contributed-power
  cost-to-active-country-of-going-to-war
  cost-to-target-of-going-to-war
  target-alliance-win-count

  total-wealth
  united-states-wealth
  russia-wealth
  canada-wealth
  united-kingdom-wealth
  france-wealth
  germany-wealth
  china-wealth
  japan-wealth
  final-output
  csv-name

]

turtles-own[
  wealth ;;the Iron and Steel Production variable (irst) from the NMC data is used as a proxy for wealth
  logwealth ;;for a plot on the interface
  milex_percap ;;military expenditures per capita
  StateNme ;;the name of the Country
  cinc ;;composite index of national capabilities
  cincAdj ;;Resized CINC to make the largest value in each year get a value of 1
  targets ;;the lis of targets that an Active agent could choose from
  vulnerability ;;calculated as (Wa - Wt)/Wa where Wa and Wt are the wealths of the active actor and the target, respectively.
  payment ;;the amount a country would have to pay in tribute to another country if they were targeted and did not want to go to war (equal to half of wealth)
  attack-score ;; the product of vulnerability and payment
  year
  contributed-power
  targets-allliance ;;the targets Alliance that will aid them if attacked
  targets-allliance-contribution
]

links-own[
  AllianceLevel
  YearsAllied
]



to runOneYear
  reset-variables

  ask turtles [

    set wealth (wealth + (wealth * 0.01)) ;;increase the global wealth for all countries by a certain percentage

    ;if wealth > 0 [set logwealth ln(wealth)] ;;for a plot on the interface
    ]

  set num-active-countries-each-year 23 ;int (count turtles - (count turtles * .7)) ;30% of the total number of agents are asked to be active each year
  repeat num-active-countries-each-year [ ;;each year the specified number of countries are active
    reset-variables
    ask turtles [set color green] ;;return all colors to the original green color
    set active-country one-of turtles ;;any turtle can be active
    ask active-country[
      ifelse wealth > 0 [ ;;a country can only decide to become agressive if the have the means to do so. in this model wealth must be positive to even consider attacking
        set color red
        set targets other turtles with [not in-link-neighbor? active-country];the Active country can attack another country as long as it is not connected to that country with a defense agreement
        decide-to-target ;;uses the list of targets for the active-country to, as Axtell says: "choose among the potential targets the one that maximizes the product of the target's vulnerability times its possible payment"
        decide-to-war ;;impletments the war code
      ][]
    ]
  ]
  ask turtles[

    if wealth > 0 [set size sqrt sqrt sqrt (wealth) ] ;; scale down the wealth and resize the nodes
  ]
  update-numbers
if narrative?[
 if cost-to-active-country-of-going-to-war > cost-to-target-of-going-to-war [print "###active-country was dealt more damage becuase of targets Alliance" set target-alliance-win-count (target-alliance-win-count + 1)]
  ]
  if output-csv?[
    prepare-behavior-space-output
    write-csv csv-name final-output]
  tick
  if layout? [layout]
  if (ticks = 100) [stop] ;;stop after 1000 ticks
end

to decide-to-war
  if current-target != nobody [
    if narrative? [print "The country of" ask active-country [print StateNme] print "decided to target" ask current-target [print StateNme]]
    ;    ask current-target[set payment wealth * tribute-rate]
    ask current-target[set payment wealth * tribute-rate]
    let current-targetTribute [payment] of current-target ;make local variable for the payment the active country would receive from the target if there is not a war
    let current-targetWealth [wealth] of current-target ;;local variable for wealth of current-target
    set cost-to-target-of-going-to-war (0.25 * wealth) ;the damage to the target is equal to 25% of the wealth of the active-country
    ifelse [targets-allliance-contribution] of current-target > 0 [
      set cost-to-active-country-of-going-to-war (0.25 * ([targets-allliance-contribution] of current-target + current-targetWealth))
    ][
      set cost-to-active-country-of-going-to-war (0.25 * current-targetWealth)
    ] ;the damage to the active-country is equal to 25% of the wealth of the targeted country

    ; only go to war if you have the money to go to war
    if wealth >= cost-to-active-country-of-going-to-war and random-float 1 <= democratic-efficiency[
    ;;;If cost of going to war is less that cost of paying tribute to the Active country War will hapen...


      ifelse (cost-to-target-of-going-to-war < current-targetTribute) [
        go-to-war
  ]

  ;;;;If the cost of going to war is greater than just paying the tribute then the targeted country may pay tribute
  ;;;;If the random number between 0 and 1 is greater than the rate of democratic efficiency and the nation can afford to go to war,
      ; defending nation will still go to war even if it expects to lose.
  [ifelse random-float 1 > democratic-efficiency and current-targetWealth > cost-to-target-of-going-to-war
        [go-to-war]

        ;; paying tribute is the 'rational' decision (random number is less than the democratic-efficiency)
        [pay-tribute current-targetTribute]
      ]

   ;  ifelse random-float 1 > democratic-efficiency

    ]
  ]

end
to go-to-war
  ;;;;This is War:
      ;;my wealth is decreased in proportion to the wealth of the country I attacked
      set wealth (wealth - cost-to-active-country-of-going-to-war)
      ;;the targets wealth is decreased in proportion to the wealth of the active-country
      ask current-target [
        set wealth (wealth - cost-to-target-of-going-to-war)
        if is-agentset? targets-allliance AND any? targets-allliance [
          ask targets-allliance [set wealth (wealth - contributed-power)]
        ]
      ]
      set number-of-wars (number-of-wars + 1) ;;keep track f the number of times a War happens
      if narrative? [print "War broke out & active-country inflicted damage of:" show cost-to-target-of-going-to-war print "targetCountry inflicted damage of:" show cost-to-active-country-of-going-to-war]
end

to pay-tribute[current-targetTribute]
ask current-target [
        ; Don't allow wealth to drop below 0
        if wealth < current-targetTribute[set current-targetTribute wealth * .25]
        set wealth (wealth - current-targetTribute)] ;;the targeted Countrys wealth is decreased by the tribute amount
    set wealth (wealth + current-targetTribute) ;the Active Countries wealth is increased by the tribute amount
    set num-tributes (num-tributes + 1) ;;keep track of the number of times a Tribute is payed
    create-link-with current-target [if not showLinks? [hide-link]]
    create-link-with current-target [if not showLinks? [hide-link]]
    if narrative? [print "The target payed a tribute of" show current-targetTribute]
end

to decide-to-target
  ask targets [
    set payment (wealth * tribute-rate)

    ifelse (wealth > 0) and (count my-in-links > 0) [ ;;a country must have at least a little wealth to be considered as a target
      calculate-alliance-help
      set vulnerability ((([wealth] of active-country) - (wealth + targets-allliance-contribution)) / ([wealth] of active-country))
            set attack-score (vulnerability * payment)
      ]
    [if wealth > 0 [
      set vulnerability (([wealth] of active-country - wealth) / [wealth] of active-country) ;;calculate vulnerability from Axtell: (Wa - Wt)/Wa where Wa and Wt are the wealths of the active actor and the target, respectively.
      set attack-score (vulnerability * payment) ;;calculate attack score which will be used to chose the optimal target for the attacking country
      ]
    ]
  ]

  ; add randomness to choice of target
;  ifelse random-float 1 > democratic-efficiency
;    [set current-target one-of targets
;
;      if current-target != nobody [ask current-target [
;        set color orange
;        if (wealth > 0) and (count my-in-links > 0) [ask targets-allliance [set color yellow]]
;        ]
;      ]
;  ]
  set current-target one-of targets with-max [attack-score] ;;find the country with the maximum attack score and make them the current-target. if more than one country is tied for highest score then only one is chosen

    if current-target != nobody [ask current-target [
      set color orange
      if (wealth > 0) and (count my-in-links > 0) [ask targets-allliance [set color yellow]]
      ]
    ]


  ;;;;;;;;;;;;;;;;;;
  ;; CATON ADDITION
  ; Sometimes democracies choose to fight wars where the cost is greater than the benefit

  ;;;;;;;;;;;;;;;;;;

end

to calculate-alliance-help ;;this is a target specific routine
  set targets-allliance other turtles with [in-link-neighbor? myself] ;create turtle set of countries that will help the target if there is war
  ;if narrative? [print "###################################################"
    ;print "the country gets help from other countries" show targets-allliance]
  ask targets-allliance [
    if (not out-link-neighbor? active-country) AND (out-link-neighbor? myself)[
      set contributed-power (wealth * ([AllianceLevel] of out-link-to myself)) ;ask the targets alliance to create a list of the amount they are willing to contribute
      ;if narrative? [print "the countries contribute the following amounts:" show contributed-power]
    ]
  ]
  set targets-allliance-contribution sum [contributed-power] of targets-allliance
  ;if narrative? [show targets-allliance-contribution]


end


to setupYearGraph
  ca
  reset-ticks

  ;if Year = 1945[nw:load-graphml "C:/Users/Harold/Documents/CSS/CSS610/Global Actors Research/Data for NetLogo Model/1945.graphml"]
  ;if Year = 1960[nw:load-graphml LocationOfGraph]
  nw:load-graphml LocationOfGraph ;should be a graphml file
  set global-wealth (sum [wealth] of turtles) ;;update global wealth variable
  set last-year-wealth global-wealth


  ask turtles [
    set color green
    set shape "dot"
    set size sqrt sqrt sqrt wealth;resize the nodes on wealth or power
  ]

  set number-of-wars 0
  set num-tributes 0
  set target-alliance-win-count 0
  prep-csv-name
  layout
end


to layout
  ;; the number 10 here is arbitrary; more repetitions slows down the
  ;; model, but too few gives poor layouts
  repeat 5 [
    do-layout
    display  ;; so we get smooth animation
  ]
end

to do-layout
  repeat 100 [layout-spring turtles links 0.2 4 0.9]
end

to reset-variables
  ask turtles [
    set targets 0
    set vulnerability 0
    set payment 0
    set attack-score 0
    set contributed-power 0
    set targets-allliance 0
    set targets-allliance-contribution 0
  ]
end

to update-numbers
  if (num-tributes > 0)[set war-to-tribute-ratio (number-of-wars / num-tributes)]

  set global-wealth (sum [wealth] of turtles) ;;update global wealth variable
 ; set global-wealth-change (ln (global-wealth / last-year-wealth)) * 100
  set last-year-wealth global-wealth
  set united-states-wealth item 0 [wealth] of turtles with [stateNme = "United States of America"]
  set russia-wealth item 0 [wealth] of turtles with [stateNme = "Russia"]
  set canada-wealth ifelse-value (any? turtles with [stateNme ="Canada"])[ item 0 [wealth] of turtles with [stateNme = "Canada"]][0]
  set united-kingdom-wealth  item 0 [wealth] of turtles with [stateNme = "United Kingdom"]
  set france-wealth  item 0 [wealth] of turtles with [stateNme = "France"]

  set germany-wealth ifelse-value (any? turtles with [stateNme ="Germany"])[ item 0 [wealth] of turtles with [stateNme = "Germany"]][0]
  set china-wealth  item 0 [wealth] of turtles with [stateNme = "China"]
  set japan-wealth  item 0 [wealth] of turtles with [stateNme = "Japan"]


  ;tick
end




to setup-random
  clear-all-plots
  reset-ticks
  ask links [set AllianceLevel 0.0 + random-float 1.0]
  ask turtles [set wealth 300 + random 200]


end


to prepare-behavior-space-output


  set final-output (list
    tribute-rate
    democratic-efficiency
    global-wealth
    number-of-wars
    num-tributes
    war-to-tribute-ratio
    united-states-wealth
    russia-wealth
    canada-wealth
    united-kingdom-wealth
    france-wealth
    germany-wealth
    china-wealth
    japan-wealth
)
end

to write-csv [ #filename #items ]
  ;; #items is a list of the data (or headers!) to write.
  if is-list? #items and not empty? #items
  [ file-open #filename
    ;; quote non-numeric items
    set #items map quote #items
    ;; print the items
    ;; if only one item, print it.

    foreach but-last #items
      [ x -> file-type(word x "," )]
;        file-print reduce [ (word item i #items "," ) ] #items
    file-print last #items


    ;; close-up
    file-close
  ]
end

to prep-csv-name
;  set csv-name "0GA1960Tribute30Percent.csv"
;  set csv-name "0GA1945DemEfficiency90PercentPlus.csv"
;set csv-name "0GA1945TenByTen.csv"
  set csv-name"0GA1960100By100.csv"
  ;  set csv-name "0test.csv"
  set csv-name replace-item 0 csv-name (word behaviorspace-run-number)

end

to-report quote [ #thing ]
  ifelse is-number? #thing
  [ report #thing ]
  [ report (word "\"" #thing "\"") ]
end
@#$#@#$#@
GRAPHICS-WINDOW
217
10
721
515
-1
-1
12.1
1
10
1
1
1
0
0
0
1
-20
20
-20
20
0
0
1
ticks
30.0

BUTTON
5
15
68
48
setup
setupYearGraph
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
4
71
102
104
runOneYear
runOneYear
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
107
72
205
105
runOneYear
runOneYear
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

PLOT
727
11
927
161
Number of Wars and Tributes
time
num
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -2674135 true "" "plot number-of-wars"
"pen-1" 1.0 0 -7500403 true "" "plot num-tributes"

PLOT
726
171
926
321
War/Tribute Ratio
time
War/Tribute Ratio
0.0
10.0
0.0
1.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot war-to-tribute-ratio"

MONITOR
807
188
919
233
War/Tribute Ratio
war-to-tribute-ratio
3
1
11

SWITCH
6
195
118
228
narrative?
narrative?
1
1
-1000

SLIDER
4
114
176
147
tribute-rate
tribute-rate
0
1
1.0
.1
1
NIL
HORIZONTAL

PLOT
936
13
1136
163
plot countries wealth
countries
wealth
0.0
10.0
0.0
10.0
true
false
"" "if ticks > 0 [ set-histogram-num-bars count (turtles with [wealth > 0])]"
PENS
"default" 1.0 1 -16777216 true "" "histogram [logwealth] of turtles with [logwealth > 0]"

SWITCH
6
233
109
266
layout?
layout?
1
1
-1000

MONITOR
1139
13
1264
58
Target Alliance Wins
target-alliance-win-count
0
1
11

SWITCH
6
157
123
190
showLinks?
showLinks?
1
1
-1000

SLIDER
5
437
177
470
num-nodes
num-nodes
0
200
100.0
1
1
NIL
HORIZONTAL

INPUTBOX
725
455
1297
515
LocationOfGraph
C:\\Users\\JLCat\\OneDrive\\Documents\\Network Files\\1960defense.graphml
1
0
String

BUTTON
954
341
1042
374
loadGraph
nw:load-graphml LocationOfGraph
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

MONITOR
1158
80
1252
125
Num Countries
count turtles
17
1
11

MONITOR
1166
145
1223
190
Year
max [year] of turtles
17
1
11

SLIDER
3
271
175
304
democratic-efficiency
democratic-efficiency
0
1
1.0
.01
1
NIL
HORIZONTAL

SWITCH
17
333
138
366
output-csv?
output-csv?
0
1
-1000

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.0.2
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="experiment .01-1 DE" repetitions="100" runMetricsEveryStep="true">
    <setup>setupYearGraph</setup>
    <go>runOneYear</go>
    <enumeratedValueSet variable="LocationOfGraph">
      <value value="&quot;C:\\Users\\JLCat\\OneDrive\\Documents\\Network Files\\1960defense.graphml&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="output-csv?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="showLinks?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="layout?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-nodes">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="narrative?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="tribute-rate">
      <value value="0.3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="democratic-efficiency">
      <value value="0.01"/>
      <value value="0.02"/>
      <value value="0.03"/>
      <value value="0.04"/>
      <value value="0.05"/>
      <value value="0.06"/>
      <value value="0.07"/>
      <value value="0.08"/>
      <value value="0.09"/>
      <value value="0.1"/>
      <value value="0.11"/>
      <value value="0.12"/>
      <value value="0.13"/>
      <value value="0.14"/>
      <value value="0.15"/>
      <value value="0.16"/>
      <value value="0.17"/>
      <value value="0.18"/>
      <value value="0.19"/>
      <value value="0.2"/>
      <value value="0.21"/>
      <value value="0.22"/>
      <value value="0.23"/>
      <value value="0.24"/>
      <value value="0.25"/>
      <value value="0.26"/>
      <value value="0.27"/>
      <value value="0.28"/>
      <value value="0.29"/>
      <value value="0.3"/>
      <value value="0.31"/>
      <value value="0.32"/>
      <value value="0.33"/>
      <value value="0.34"/>
      <value value="0.35"/>
      <value value="0.36"/>
      <value value="0.37"/>
      <value value="0.38"/>
      <value value="0.39"/>
      <value value="0.4"/>
      <value value="0.41"/>
      <value value="0.42"/>
      <value value="0.43"/>
      <value value="0.44"/>
      <value value="0.45"/>
      <value value="0.46"/>
      <value value="0.47"/>
      <value value="0.48"/>
      <value value="0.49"/>
      <value value="0.5"/>
      <value value="0.51"/>
      <value value="0.52"/>
      <value value="0.53"/>
      <value value="0.54"/>
      <value value="0.55"/>
      <value value="0.56"/>
      <value value="0.57"/>
      <value value="0.58"/>
      <value value="0.59"/>
      <value value="0.6"/>
      <value value="0.61"/>
      <value value="0.62"/>
      <value value="0.63"/>
      <value value="0.64"/>
      <value value="0.65"/>
      <value value="0.66"/>
      <value value="0.67"/>
      <value value="0.68"/>
      <value value="0.69"/>
      <value value="0.7"/>
      <value value="0.71"/>
      <value value="0.72"/>
      <value value="0.73"/>
      <value value="0.74"/>
      <value value="0.75"/>
      <value value="0.76"/>
      <value value="0.77"/>
      <value value="0.78"/>
      <value value="0.79"/>
      <value value="0.8"/>
      <value value="0.81"/>
      <value value="0.82"/>
      <value value="0.83"/>
      <value value="0.84"/>
      <value value="0.85"/>
      <value value="0.86"/>
      <value value="0.87"/>
      <value value="0.88"/>
      <value value="0.89"/>
      <value value="0.9"/>
      <value value="0.91"/>
      <value value="0.92"/>
      <value value="0.93"/>
      <value value="0.94"/>
      <value value="0.95"/>
      <value value="0.96"/>
      <value value="0.97"/>
      <value value="0.98"/>
      <value value="0.99"/>
      <value value="1"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="experiment .9-1 DE" repetitions="100" runMetricsEveryStep="true">
    <setup>setupYearGraph</setup>
    <go>runOneYear</go>
    <enumeratedValueSet variable="LocationOfGraph">
      <value value="&quot;C:\\Users\\JLCat\\OneDrive\\Documents\\Network Files\\1945defense.graphml&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="output-csv?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="showLinks?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="layout?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-nodes">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="narrative?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="tribute-rate">
      <value value="0.3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="democratic-efficiency">
      <value value="0.901"/>
      <value value="0.902"/>
      <value value="0.903"/>
      <value value="0.904"/>
      <value value="0.905"/>
      <value value="0.906"/>
      <value value="0.907"/>
      <value value="0.908"/>
      <value value="0.909"/>
      <value value="0.91"/>
      <value value="0.911"/>
      <value value="0.912"/>
      <value value="0.913"/>
      <value value="0.914"/>
      <value value="0.915"/>
      <value value="0.916"/>
      <value value="0.917"/>
      <value value="0.918"/>
      <value value="0.919"/>
      <value value="0.92"/>
      <value value="0.921"/>
      <value value="0.922"/>
      <value value="0.923"/>
      <value value="0.924"/>
      <value value="0.925"/>
      <value value="0.926"/>
      <value value="0.927"/>
      <value value="0.928"/>
      <value value="0.929"/>
      <value value="0.93"/>
      <value value="0.931"/>
      <value value="0.932"/>
      <value value="0.933"/>
      <value value="0.934"/>
      <value value="0.935"/>
      <value value="0.936"/>
      <value value="0.937"/>
      <value value="0.938"/>
      <value value="0.939"/>
      <value value="0.94"/>
      <value value="0.941"/>
      <value value="0.942"/>
      <value value="0.943"/>
      <value value="0.944"/>
      <value value="0.945"/>
      <value value="0.946"/>
      <value value="0.947"/>
      <value value="0.948"/>
      <value value="0.949"/>
      <value value="0.95"/>
      <value value="0.951"/>
      <value value="0.952"/>
      <value value="0.953"/>
      <value value="0.954"/>
      <value value="0.955"/>
      <value value="0.956"/>
      <value value="0.957"/>
      <value value="0.958"/>
      <value value="0.959"/>
      <value value="0.96"/>
      <value value="0.961"/>
      <value value="0.962"/>
      <value value="0.963"/>
      <value value="0.964"/>
      <value value="0.965"/>
      <value value="0.966"/>
      <value value="0.967"/>
      <value value="0.968"/>
      <value value="0.969"/>
      <value value="0.97"/>
      <value value="0.971"/>
      <value value="0.972"/>
      <value value="0.973"/>
      <value value="0.974"/>
      <value value="0.975"/>
      <value value="0.976"/>
      <value value="0.977"/>
      <value value="0.978"/>
      <value value="0.979"/>
      <value value="0.98"/>
      <value value="0.981"/>
      <value value="0.982"/>
      <value value="0.983"/>
      <value value="0.984"/>
      <value value="0.985"/>
      <value value="0.986"/>
      <value value="0.987"/>
      <value value="0.988"/>
      <value value="0.989"/>
      <value value="0.99"/>
      <value value="0.991"/>
      <value value="0.992"/>
      <value value="0.993"/>
      <value value="0.994"/>
      <value value="0.995"/>
      <value value="0.996"/>
      <value value="0.997"/>
      <value value="0.998"/>
      <value value="0.999"/>
      <value value="1"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="experiment" repetitions="5" runMetricsEveryStep="true">
    <setup>setupYearGraph</setup>
    <go>runOneYear</go>
    <enumeratedValueSet variable="LocationOfGraph">
      <value value="&quot;C:\\Users\\JLCat\\OneDrive\\Documents\\Network Files\\1980defense.graphml&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="output-csv?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="showLinks?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="layout?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-nodes">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="narrative?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="tribute-rate">
      <value value="0.1"/>
      <value value="0.5"/>
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="democratic-efficiency">
      <value value="0.1"/>
      <value value="0.5"/>
      <value value="1"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="experiment .01-1 TR and DE" repetitions="20" runMetricsEveryStep="true">
    <setup>setupYearGraph</setup>
    <go>runOneYear</go>
    <enumeratedValueSet variable="LocationOfGraph">
      <value value="&quot;C:\\Users\\JLCat\\OneDrive\\Documents\\Network Files\\1960defense.graphml&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="output-csv?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="showLinks?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="layout?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-nodes">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="narrative?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="tribute-rate">
      <value value="0.01"/>
      <value value="0.02"/>
      <value value="0.03"/>
      <value value="0.04"/>
      <value value="0.05"/>
      <value value="0.06"/>
      <value value="0.07"/>
      <value value="0.08"/>
      <value value="0.09"/>
      <value value="0.1"/>
      <value value="0.11"/>
      <value value="0.12"/>
      <value value="0.13"/>
      <value value="0.14"/>
      <value value="0.15"/>
      <value value="0.16"/>
      <value value="0.17"/>
      <value value="0.18"/>
      <value value="0.19"/>
      <value value="0.2"/>
      <value value="0.21"/>
      <value value="0.22"/>
      <value value="0.23"/>
      <value value="0.24"/>
      <value value="0.25"/>
      <value value="0.26"/>
      <value value="0.27"/>
      <value value="0.28"/>
      <value value="0.29"/>
      <value value="0.3"/>
      <value value="0.31"/>
      <value value="0.32"/>
      <value value="0.33"/>
      <value value="0.34"/>
      <value value="0.35"/>
      <value value="0.36"/>
      <value value="0.37"/>
      <value value="0.38"/>
      <value value="0.39"/>
      <value value="0.4"/>
      <value value="0.41"/>
      <value value="0.42"/>
      <value value="0.43"/>
      <value value="0.44"/>
      <value value="0.45"/>
      <value value="0.46"/>
      <value value="0.47"/>
      <value value="0.48"/>
      <value value="0.49"/>
      <value value="0.5"/>
      <value value="0.51"/>
      <value value="0.52"/>
      <value value="0.53"/>
      <value value="0.54"/>
      <value value="0.55"/>
      <value value="0.56"/>
      <value value="0.57"/>
      <value value="0.58"/>
      <value value="0.59"/>
      <value value="0.6"/>
      <value value="0.61"/>
      <value value="0.62"/>
      <value value="0.63"/>
      <value value="0.64"/>
      <value value="0.65"/>
      <value value="0.66"/>
      <value value="0.67"/>
      <value value="0.68"/>
      <value value="0.69"/>
      <value value="0.7"/>
      <value value="0.71"/>
      <value value="0.72"/>
      <value value="0.73"/>
      <value value="0.74"/>
      <value value="0.75"/>
      <value value="0.76"/>
      <value value="0.77"/>
      <value value="0.78"/>
      <value value="0.79"/>
      <value value="0.8"/>
      <value value="0.81"/>
      <value value="0.82"/>
      <value value="0.83"/>
      <value value="0.84"/>
      <value value="0.85"/>
      <value value="0.86"/>
      <value value="0.87"/>
      <value value="0.88"/>
      <value value="0.89"/>
      <value value="0.9"/>
      <value value="0.91"/>
      <value value="0.92"/>
      <value value="0.93"/>
      <value value="0.94"/>
      <value value="0.95"/>
      <value value="0.96"/>
      <value value="0.97"/>
      <value value="0.98"/>
      <value value="0.99"/>
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="democratic-efficiency">
      <value value="0.01"/>
      <value value="0.02"/>
      <value value="0.03"/>
      <value value="0.04"/>
      <value value="0.05"/>
      <value value="0.06"/>
      <value value="0.07"/>
      <value value="0.08"/>
      <value value="0.09"/>
      <value value="0.1"/>
      <value value="0.11"/>
      <value value="0.12"/>
      <value value="0.13"/>
      <value value="0.14"/>
      <value value="0.15"/>
      <value value="0.16"/>
      <value value="0.17"/>
      <value value="0.18"/>
      <value value="0.19"/>
      <value value="0.2"/>
      <value value="0.21"/>
      <value value="0.22"/>
      <value value="0.23"/>
      <value value="0.24"/>
      <value value="0.25"/>
      <value value="0.26"/>
      <value value="0.27"/>
      <value value="0.28"/>
      <value value="0.29"/>
      <value value="0.3"/>
      <value value="0.31"/>
      <value value="0.32"/>
      <value value="0.33"/>
      <value value="0.34"/>
      <value value="0.35"/>
      <value value="0.36"/>
      <value value="0.37"/>
      <value value="0.38"/>
      <value value="0.39"/>
      <value value="0.4"/>
      <value value="0.41"/>
      <value value="0.42"/>
      <value value="0.43"/>
      <value value="0.44"/>
      <value value="0.45"/>
      <value value="0.46"/>
      <value value="0.47"/>
      <value value="0.48"/>
      <value value="0.49"/>
      <value value="0.5"/>
      <value value="0.51"/>
      <value value="0.52"/>
      <value value="0.53"/>
      <value value="0.54"/>
      <value value="0.55"/>
      <value value="0.56"/>
      <value value="0.57"/>
      <value value="0.58"/>
      <value value="0.59"/>
      <value value="0.6"/>
      <value value="0.61"/>
      <value value="0.62"/>
      <value value="0.63"/>
      <value value="0.64"/>
      <value value="0.65"/>
      <value value="0.66"/>
      <value value="0.67"/>
      <value value="0.68"/>
      <value value="0.69"/>
      <value value="0.7"/>
      <value value="0.71"/>
      <value value="0.72"/>
      <value value="0.73"/>
      <value value="0.74"/>
      <value value="0.75"/>
      <value value="0.76"/>
      <value value="0.77"/>
      <value value="0.78"/>
      <value value="0.79"/>
      <value value="0.8"/>
      <value value="0.81"/>
      <value value="0.82"/>
      <value value="0.83"/>
      <value value="0.84"/>
      <value value="0.85"/>
      <value value="0.86"/>
      <value value="0.87"/>
      <value value="0.88"/>
      <value value="0.89"/>
      <value value="0.9"/>
      <value value="0.91"/>
      <value value="0.92"/>
      <value value="0.93"/>
      <value value="0.94"/>
      <value value="0.95"/>
      <value value="0.96"/>
      <value value="0.97"/>
      <value value="0.98"/>
      <value value="0.99"/>
      <value value="1"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="experiment TenByTen" repetitions="100" runMetricsEveryStep="true">
    <setup>setupYearGraph</setup>
    <go>runOneYear</go>
    <enumeratedValueSet variable="LocationOfGraph">
      <value value="&quot;C:\\Users\\JLCat\\OneDrive\\Documents\\Network Files\\1945defense.graphml&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="output-csv?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="showLinks?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="layout?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-nodes">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="narrative?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="tribute-rate">
      <value value="0.1"/>
      <value value="0.2"/>
      <value value="0.3"/>
      <value value="0.4"/>
      <value value="0.5"/>
      <value value="0.6"/>
      <value value="0.7"/>
      <value value="0.8"/>
      <value value="0.9"/>
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="democratic-efficiency">
      <value value="0.1"/>
      <value value="0.2"/>
      <value value="0.3"/>
      <value value="0.4"/>
      <value value="0.5"/>
      <value value="0.6"/>
      <value value="0.7"/>
      <value value="0.8"/>
      <value value="0.9"/>
      <value value="1"/>
    </enumeratedValueSet>
  </experiment>
</experiments>
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@

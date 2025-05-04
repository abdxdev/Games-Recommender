# Steam Dataset

### Recommendations Dataset

```bash
> python -c "import pandas as pd; print(pd.read_csv('dataset/recommendations.csv', nrows=3).to_string())"

    app_id  helpful  funny        date  is_recommended  hours  user_id  review_id
0   975370        0      0  2022-12-12            True   36.3    51580          0
1   304390        4      0  2017-02-17           False   11.5     2586          1
2  1085660        2      0  2019-11-17            True  336.5   253880          2
```

### Games Dataset

```bash
> python -c "import pandas as pd; print(pd.read_csv('dataset/games.csv', nrows=3).to_string())"

   app_id                              title date_release   win    mac  linux         rating  positive_ratio  user_reviews  price_final  price_original  discount  steam_deck
0   13500  Prince of Persia: Warrior Within™   2008-11-21  True  False  False  Very Positive              84          2199         9.99            9.99       0.0        True
1   22364            BRINK: Agents of Change   2011-08-03  True  False  False       Positive              85            21         2.99            2.99       0.0        True
2  113020       Monaco: What's Yours Is Mine   2013-04-24  True   True   True  Very Positive              92          3722        14.99           14.99       0.0        True
```

### Users Dataset

```bash
> python -c "import pandas as pd; print(pd.read_csv('dataset/users.csv', nrows=3).to_string())"

    user_id  products  reviews
0   7360263       359        0
1  14020781       156        1
2   8762579       329        4
```

### Games Metadata Dataset

```bash
> python -c "with open('dataset/games_metadata.json', 'r', encoding='utf-8') as f: print(f.readline())"

{"app_id":13500,"description":"Enter the dark underworld of Prince of Persia Warrior Within, the sword-slashing sequel to the critically acclaimed Prince of Persia: The Sands of Time™. Hunted by Dahaka, an immortal incarnation of Fate seeking divine retribution, the Prince embarks upon a path of both carnage and mystery to defy his preordained death.","tags":["Action","Adventure","Parkour","Third Person","Great Soundtrack","Singleplayer","Platformer","Time Travel","Atmospheric","Classic","Hack and Slash","Time Manipulation","Gore","Fantasy","Story Rich","Dark","Open World","Controller","Dark Fantasy","Puzzle"]}

```

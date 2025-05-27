# How did I setup the evaluation env

```bash
mamba create -n cobar-evaluation python=3.12
mamba activate cobar-evaluation

git clone https://github.com/NeLy-EPFL/flygym
cd flygym
git checkout cobar
pip install -e .[dev,examples]
cd ..
```

Now we need 7 more random seeds so I went to [random.org](https://www.random.org/) and generated random numbers between 0 and 20000000:

```text
19937347, 7794536, 8451236, 12868265, 16710355, 15656979, 12062889
```

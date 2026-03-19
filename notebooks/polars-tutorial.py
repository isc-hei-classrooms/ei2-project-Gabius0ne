import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A Polars Tutorial

    This tutorial will guide you through the basics of using [Polars](https://pola.rs/), a (very) fast DataFrame library for Python (written in Rust).

    DataFrames are a common data structure used in data analysis, similar to SQL tables or (to a lesser extent) Excel spreadsheets. They consist of rows and columns, where each column has a name and a data type.

    This notebook shows the basic usage of Polars, geared towards users familiar with Pandas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting things up

    Let's first install Polars and all the libraries we need for this tutorial. You can run the following command in your terminal:

    ```bash
    uv sync
    ```

    It uses [uv](https://docs.astral.sh/uv/getting-started/first-steps/), another amazing tool written in Rust, to install the required packages.

    We will now import Polars and print its version to ensure everything is set up correctly. We also import Pandas as this tutorial will occasionally [compare Polars with Pandas](https://docs.pola.rs/user-guide/migration/pandas/).
    """)
    return


@app.cell
def _():
    import polars as pl
    import pandas as pd

    print("Polars version:", pl.__version__)
    print("Pandas version:", pd.__version__)
    return pd, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's also set up [ruff](https://docs.astral.sh/ruff/), yet another Astral tool, to lint and format our code. That way, you never have to worry about formatting your code again. By using it, you agree to cede control over minutiae of hand-formatting. In return, Ruff gives you speed, determinism, and freedom from your instructor nagging about formatting. You will save time and mental energy for more important matters.

    If you're running this notebook in VS Code, you can install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) to automatically format your code as you type.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## My first Polars DataFrame

    A Polars dataframe is similar to a Pandas dataframe, but it is designed to be faster and more memory-efficient. You can create a Polars dataframe from a dictionary, a list of dictionaries, or a CSV file.
    """)
    return


@app.cell
def _(pl):
    from datetime import datetime, date

    data = {
        "a": [1, 2, 4],
        "b": [2.0, 3.0, 5.0],
        "c": ["string1", "string2", "string3"],
        "d": [date(2000, 1, 1), date(2000, 2, 1), date(2000, 3, 1)],
        "e": [
            datetime(2000, 1, 1, 12, 0),
            datetime(2000, 1, 2, 12, 0),
            datetime(2000, 1, 3, 12, 0),
        ],
    }


    df1 = pl.DataFrame(data)

    df1
    return data, df1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how Polars infers the data types of the columns automatically, just like Pandas does. However, Polars has a lot richer set of [data types](https://docs.pola.rs/user-guide/concepts/data-types-and-structures/#appendix-full-data-types-table) than Pandas.
    """)
    return


@app.cell
def _(df1):
    df1.schema
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Furthermore, Polars almost never coerces data types silently, which is a common source of bugs in Pandas:
    """)
    return


@app.cell
def _(pl):
    try:
        pl.DataFrame(
            {"a": [1, 2.0, 4], "b": [2.0, 3.0, 5.0], "c": ["string1", "string2", "string3"]}
        )
    except Exception as e:
        print("Error:", e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pandas, on the other hand, will silently coerce column 'a' to a float:
    """)
    return


@app.cell
def _(pd):
    pd.DataFrame({'a': [1, 2.0, 4], 'b': [2.0, 3.0, 5.0], 'c': ['string1', 'string2', 'string3']}).dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you want to allow Polars to coerce data types, you can set the `strict` parameter to `False` when creating the DataFrame. This will allow Polars to automatically convert data types where possible, similar to how Pandas behaves.

    > ⚠️ That's generally a bad idea and poor developer hygiene.
    """)
    return


@app.cell
def _(pl):
    pl.DataFrame(
        {"a": [1, 2.0, 4], "b": [2.0, 3.0, 5.0], "c": ["string1", "string2", "string3"]},
        strict=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also specify the data types of the columns explicitly when creating the DataFrame. This is useful if you want to ensure that the data types are correct and avoid any potential issues with data type inference.
    """)
    return


@app.cell
def _(data, pl):
    df2 = pl.DataFrame(
        data,
        schema={
            "a": pl.Int64,
            "b": pl.Float64,
            "c": pl.Utf8,
            "d": pl.Date,
            "e": pl.Datetime,
        },
    )

    df2
    return (df2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What happens if you pass data that are not compatible with the specified data types?
    """)
    return


@app.cell
def _():
    # 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can convert Polars dataframes to many other formats, such as Pandas dataframes or a list of dictionaries, using the `to_pandas()` and `to_dicts()` methods, respectively.
    """)
    return


@app.cell
def _(df2):
    df2.to_pandas()
    return


@app.cell
def _(df2):
    df2.to_dicts()
    return


@app.cell
def _(df2):
    try:
        df2.to_numpy()
    except Exception as e:
        print(f"Error occurred: {e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ooops, this failed! This is because `to_numpy()` requires a single data type for the entire DataFrame, while Polars allows mixed data types in a single DataFrame. Ideally, we would like to select numerical columns. Let's see how to do that in Polars.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Selecting columns and inspecting data

    Polars has a very powerful and flexible API for selecting columns. You can select columns by name, by index, or by a boolean mask. You can also select multiple columns at once.
    """)
    return


@app.cell
def _(df2):
    import polars.selectors as cs

    # Select only numeric columns
    df2.select(cs.numeric()).to_numpy()
    return (cs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also exclude some columns:
    """)
    return


@app.cell
def _(cs, df2):
    df2.select(cs.exclude("c"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The core base data structures provided by Polars are dataframes and **series**. A series is a 1-dimensional homogeneous data structure. By “homogeneous” we mean that all elements inside a series have the same data type.
    """)
    return


@app.cell
def _(pl):
    pl.Series("ints", [1, 2, 3, 4, 5], dtype=pl.Int64)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It so happens that a dataframe is a collection of series, where each series represents a column in the dataframe. Those can be accessed using the column name in brackets. The `select` method will always return a dataframe, even when a single column is selected.
    """)
    return


@app.cell
def _(df2):
    type(df2["a"]), type(df2.select("a"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can inspect a dataframe in multiple ways:

    - `df.head()` and `df.tail()` return the first and last rows of the dataframe, respectively.
    - `df.shape` returns the number of rows and columns in the dataframe.
    - `df.columns` returns the names of the columns in the dataframe.


    The method `df.glimpse()` shows both the shape, the types, and the values of the first few rows of a dataframe, but formats the output differently from head. Here, each line of the output corresponds to a single column, making it easier to take inspect wider dataframes.
    """)
    return


@app.cell
def _(df2):
    df2.glimpse()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manipulating data

    Up to now, we have seen very basic manipulations. We are going to go a bit deeper into data manipulations. We are going to create a slightly more complex dataset.
    """)
    return


@app.cell
def _(pl):
    df = pl.read_csv("swiss-food.csv")
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Say you want to create a column with the type of preparation, which is currently lumped in the name, after the comma.
    """)
    return


@app.cell
def _(df, pl):
    # Create a new column with the preparation type extracted from the Name column
    df_1 = df.with_columns(preparation=pl.col('Name').str.split(',').list.get(1, null_on_oob=True))
    df_1.select('Name', 'preparation')
    return (df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's clean this up a bit by removing the trailing whitespace from the preparation column and renaming it according to the format of other columns (uppercase). Just like in SQL, any expression can also appear in the `select` method:
    """)
    return


@app.cell
def _(cs, df_1, pl):
    df_2 = df_1.select('ID', 'Name', 'Category', pl.col('preparation').str.strip_chars().alias('Preparation'), cs.contains('kcal').alias('Calories'), cs.numeric() - cs.by_name('ID') - cs.contains('kcal'))
    df_2  # Selectors support arithmetic operations
    return (df_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can use a myriad of different [expressions](https://docs.pola.rs/api/python/stable/reference/expressions/index.html) to manipulate the data in the DataFrame:
    """)
    return


@app.cell
def _(cs, df_2, pl):
    df_2.select(pl.col('Name').str.len_chars().alias('name_length'), pl.col('Preparation').str.to_uppercase().alias('PREPARATION'), cs.contains('kcal').log(base=10).alias('kcal, log10'), pl.sum_horizontal(cs.contains('Fatty acids')).alias('Fatty acids, total (g)'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For really complex transformations, you can also apply custom functions to the dataframe using the `map_elements` or `map_batches` methods:
    """)
    return


@app.cell
def _(df_2, pl):
    TAGS = ['alcoholic', 'bread', 'cereal', 'cereals', 'cold', 'dairy', 'eggs', 'fish', 'flakes', 'fruit', 'meat', 'milk', 'nuts', 'oils', 'oleaginous', 'plant', 'potatoes', 'prepared', 'products', 'seeds', 'snacks', 'sweets', 'vegetables']

    def _categories_to_tags(categories: str) -> list[str]:
        return [tag for tag in TAGS if tag in categories.lower()]
    df_2.with_columns(tags=pl.col('Category').map_elements(_categories_to_tags, return_dtype=pl.List(pl.Utf8))).head(5)
    return (TAGS,)


@app.cell
def _(TAGS, df_2, pl):
    def _categories_to_tags(categories: str) -> list[str]:
        return [tag for tag in TAGS if tag in categories.lower()]
    df_3 = df_2.with_columns(tags=pl.col('Category').map_elements(_categories_to_tags, return_dtype=pl.List(pl.Utf8)))
    return (df_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is useful when you need t do something that cannot be expressed using the built-in expressions. However, it is generally slower than using the built-in expressions, so you should use it sparingly.

    Here is how you could do the same thing 5x faster using only native Polars expressions. Obviously, this only makes sense if you are running into performance issues due to the size of the dataset.
    """)
    return


@app.cell
def _(TAGS, df_3, pl):
    # magic command not supported in marimo; please file an issue to add support
    # %%timeit -n 10
    df_3.join(df_3.select(pl.col('Category').unique()).join(pl.DataFrame({'tags': TAGS}), how='cross').filter(pl.col('Category').str.to_lowercase().str.contains(pl.col('tags'))).group_by('Category').agg(pl.col('tags')).select('Category', pl.col('tags').fill_null([])), on='Category', how='left')  # Create a cross join between unique categories and tags  # Filter the tags that are in the category  # Group by category and aggregate the tags into a list  # Provide an empty list for categories without tags
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grouping and joining data

    Just like SQL, Polars allow you to group data by one or more columns and apply aggregate functions to the groups. You can also join dataframes together using the `join` method.

    As we will see, this is where Polars really shines compared to Pandas, as it is not only much faster, but also a lot more flexible and expressive.
    """)
    return


@app.cell
def _(df_3):
    # Group by category and calculate the mean of the numeric columns
    df_3.group_by('Category').mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So far, so good. Now, let's make Polars' API shine a bit.

    Group the data by 1st level category, and compute the mean calories and the sum of the fatty acids for each group:
    """)
    return


@app.cell
def _(cs, df_3, pl):
    df_3.group_by(pl.col('Category').str.split('/').list.get(0)).agg(pl.col('Calories').mean().alias('mean_calories'), pl.sum_horizontal(cs.contains('Fatty acids')).sum().alias('total_fatty_acids'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's push this a bit further.

    Assume I am running a study of different dietary habits, and I want to measure key metrics for different regimes. For instance, I will measure:

    - The vitamin C intake.
    - The vitamin B12 intake.
    - The omega-3 to cholesterol ratio.
    - The omega-6 to omega-3 ratio.
    """)
    return


@app.cell
def _(cs, df_3, pl):
    def vitamin_c() -> pl.Expr:
        return cs.contains('Vitamin C').alias('vitamin_c')

    def vitamin_b12() -> pl.Expr:
        return cs.contains('Vitamin B12').alias('vitamin_b12')

    def omega3() -> pl.Expr:
        return (pl.col('Eicosapentaenoic acid EPA (g)') + pl.col('Docosahexaenoic acid (DHA) (g)') + pl.col('Alpha-linolenic acid (g)')).alias('omega3')

    def omega6() -> pl.Expr:
        return pl.col('Linoleic acid (g)').alias('omega6')

    def omega3_per_cholesterol() -> pl.Expr:
        return (omega3() / (pl.col('Cholesterol (mg)') + 1e-06)).alias('omega3_per_cholesterol')

    def omega6_per_omega3() -> pl.Expr:
        return (omega6() / (omega3() + 1e-06)).alias('omega6_per_omega3')
    regimes = {'vegetarian': ~pl.col('tags').list.contains('meat') & ~pl.col('tags').list.contains('fish'), 'vegan': ~pl.col('tags').list.contains('meat') & ~pl.col('tags').list.contains('fish') & ~pl.col('tags').list.contains('dairy') & ~pl.col('tags').list.contains('eggs'), 'paleo': ~pl.col('tags').list.contains('dairy') & ~pl.col('tags').list.contains('cereal') & ~pl.col('tags').list.contains('cereals') & ~pl.col('tags').list.contains('bread') & ~pl.col('tags').list.contains('prepared') & (pl.col('tags').list.contains('meat') | pl.col('tags').list.contains('fish') | pl.col('tags').list.contains('fruit') | pl.col('tags').list.contains('vegetables') | pl.col('tags').list.contains('nuts') | pl.col('tags').list.contains('seeds')), 'low_carb': ~pl.col('tags').list.contains('bread') & ~pl.col('tags').list.contains('cereal') & ~pl.col('tags').list.contains('cereals') & ~pl.col('tags').list.contains('potatoes') & ~pl.col('tags').list.contains('sweets') & ~pl.col('tags').list.contains('snacks'), 'high_protein': pl.col('tags').list.contains('meat') | pl.col('tags').list.contains('fish') | pl.col('tags').list.contains('eggs') | pl.col('tags').list.contains('dairy') | pl.col('tags').list.contains('nuts') | pl.col('tags').list.contains('seeds'), 'low_fat': ~pl.col('tags').list.contains('oils') & ~pl.col('tags').list.contains('oleaginous') & ~pl.col('tags').list.contains('nuts') & ~pl.col('tags').list.contains('seeds'), 'dairy_free': ~pl.col('tags').list.contains('dairy') & ~pl.col('tags').list.contains('milk'), 'raw': ~pl.col('tags').list.contains('prepared') & (pl.col('tags').list.contains('fruit') | pl.col('tags').list.contains('vegetables') | pl.col('tags').list.contains('nuts') | pl.col('tags').list.contains('seeds')), 'pirate': pl.col('tags').list.contains('fish') | pl.col('tags').list.contains('seafood') | pl.col('tags').list.contains('oils')}
    regime_stats = pl.concat([df_3.group_by(pl.lit(regime)).agg(vitamin_c().filter(regimes[regime]).mean(), vitamin_b12().filter(regimes[regime]).mean().alias('vitamin_b12'), omega3_per_cholesterol().filter(regimes[regime]).mean().alias('omega3_per_cholesterol'), omega6_per_omega3().filter(regimes[regime]).mean().alias('omega6_per_omega3')) for regime in regimes.keys()]).sort(by='omega3_per_cholesterol', descending=True)
    # Different regimes expressed as Polars expressions
    regime_stats
    return (
        omega3_per_cholesterol,
        omega6_per_omega3,
        regimes,
        vitamin_b12,
        vitamin_c,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Some of these values are difficult to interpret, so it would be nice to normalize them with respect to the rest of the dataset by applying a z-score normalization. This is a common technique in statistics to normalize data by subtracting the mean and dividing by the standard deviation.
    """)
    return


@app.cell
def _(
    df_3,
    omega3_per_cholesterol,
    omega6_per_omega3,
    pl,
    regimes,
    vitamin_b12,
    vitamin_c,
):
    def zscore(expr: pl.Expr, regime: str) -> pl.Expr:
        return (expr.filter(regimes[regime]).mean() - expr.mean()) / expr.std()
    regime_stats_1 = pl.concat([df_3.group_by(pl.lit(regime)).agg(zscore(vitamin_c(), regime), zscore(vitamin_b12(), regime), zscore(omega3_per_cholesterol(), regime), zscore(omega6_per_omega3(), regime)) for regime in regimes.keys()]).sort(by='vitamin_c', descending=True)
    return (regime_stats_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's try to display these metrics as nicely as possible. We will use the configuration context offered by Polars to define the styles of the output.
    """)
    return


@app.cell
def _(pl, regime_stats_1):
    with pl.Config() as cfg:
        cfg.set_float_precision(2)  # No need to have more than 2 digits of precision
        cfg.set_tbl_formatting('ASCII_MARKDOWN')
        cfg.set_tbl_hide_column_data_types(True)  # Use a Markdown representation ready to be pasted in a Markdown document
        print(regime_stats_1)  # No need to display the types
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Not surprisingly, we see that Pirates have a rather poor Vitamin C intake, which is why scorbut was such a big problem for them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## That's it!

    For a list of curated resources about Polars, check out [Awesome Polars](https://github.com/ddotta/awesome-polars).

    That's it, thanks a lot for following this tutorial!
    """)
    return


if __name__ == "__main__":
    app.run()

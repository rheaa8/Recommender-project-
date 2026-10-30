const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, Footer,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
} = require("docx");

const R = JSON.parse(fs.readFileSync("results/results.json", "utf8"));
const D = JSON.parse(fs.readFileSync("results/dims.json", "utf8"));
const F = R.final, B = R.best;
const f3 = (x) => x.toFixed(3), f4 = (x) => x.toFixed(4), pc = (x, d = 1) => (x * 100).toFixed(d) + "%";
const nf = (x) => Number(x).toLocaleString("en-US");
const FONT = "Times New Roman";
const W = 9026; // content width (A4, 1" margins)

// ---------- helpers ----------
const SUBRE = /_(\{[^}]*\}|\([^)]*\)|[A-Za-z0-9]{1,3}(?![A-Za-z0-9_.]))/g;
function subRuns(text, base) {
  const out = []; let last = 0, m; SUBRE.lastIndex = 0;
  while ((m = SUBRE.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    let t = m[1]; if (t[0] === "{" || t[0] === "(") t = t.slice(1, -1);
    out.push(new TextRun({ text: t, font: FONT, subScript: true, ...base }));
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
function runs(text, base = {}) {
  // **bold**  and  *italic*  inline markup
  const out = []; const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g; let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(...subRuns(text.slice(last, m.index), base));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(...subRuns(text.slice(last), base));
  return out;
}
const P = (text, o = {}) => new Paragraph({
  alignment: o.align || AlignmentType.JUSTIFIED, spacing: { after: o.after ?? 120, line: o.line ?? 300, before: o.before ?? 0 },
  indent: o.indent, keepNext: o.keepNext, children: runs(text, { size: o.size ?? 23, ...(o.run || {}) }),
});
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 }, children: [new TextRun({ text: t, font: FONT })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 100 }, children: [new TextRun({ text: t, font: FONT })] });
const bullet = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 70, line: 290 }, children: runs(t, { size: 23 }) });
const num = (t) => new Paragraph({ numbering: { reference: "num", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 90, line: 290 }, children: runs(t, { size: 23 }) });
const eq = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 }, children: runs(t, { italics: true, size: 23 }) });
const cap = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 }, children: runs(t, { size: 20, italics: true }) });
const tcap = (t) => new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80 }, children: runs(t, { size: 20, bold: true }) });
const pb = () => new Paragraph({ children: [new PageBreak()] });

function fig(name, widthPx = 600) {
  const [w, h] = D[name];
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(`results/figs/${name}.png`), transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
      altText: { title: name, description: name, name } })] });
}

const bd = { style: BorderStyle.SINGLE, size: 4, color: "9AA5B1" };
const borders = { top: bd, bottom: bd, left: bd, right: bd };
function table(widths, rows, o = {}) {
  const sz = o.size ?? 19;
  const tr = rows.map((r, ri) => new TableRow({ tableHeader: ri === 0, cantSplit: true,
    children: r.map((c, ci) => {
      const isH = ri === 0; const bold = isH || (o.boldRow && o.boldRow.includes(ri));
      const al = (o.left || [0]).includes(ci) ? AlignmentType.LEFT : AlignmentType.CENTER;
      return new TableCell({ borders, width: { size: widths[ci], type: WidthType.DXA },
        shading: { type: ShadingType.CLEAR, fill: isH ? "1F3864" : (o.hl && o.hl.includes(ri) ? "E2EFDA" : (ri % 2 ? "FFFFFF" : "F5F7FA")), color: "auto" },
        margins: { top: 50, bottom: 50, left: 90, right: 90 },
        children: String(c).split("\n").map((line) => new Paragraph({ alignment: al, spacing: { after: 0, line: 250 },
          children: runs(line, { size: sz, bold, color: isH ? "FFFFFF" : undefined }) })) });
    }) }));
  return new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths, rows: tr });
}

// ---------- numbers pulled from results ----------
const order = ["Random", "Popularity", "Bias baseline", "Content-based (genres)", "User-KNN", "Item-KNN", "SVD (residual)",
  "ALS Matrix Factorisation", "Item-KNN (implicit)", "PureSVD (binary)", "Hybrid (PureSVD+KNN+Content)"];
const ratingModels = ["Bias baseline", "User-KNN", "Item-KNN", "SVD (residual)", "ALS Matrix Factorisation"];
const bestRating = ratingModels.reduce((a, b) => (F[a].RMSE < F[b].RMSE ? a : b));
const hyb = F["Hybrid (PureSVD+KNN+Content)"], psv = F["PureSVD (binary)"], iki = F["Item-KNN (implicit)"], pop = F["Popularity"];
const hist = R.als_history; const hl = hist[hist.length - 1];
const seg = R.segments; const sn = seg.ndcg;
const lift = (s) => (sn["Hybrid (PureSVD+KNN+Content)"][s] / sn["Popularity"][s]).toFixed(1);
const gain = ((F["Bias baseline"].RMSE - F[bestRating].RMSE) / F["Bias baseline"].RMSE * 100).toFixed(1);
const hv = R.tune.Hybrid; const hvGet = (w) => hv.find((d) => JSON.stringify(d.w) === JSON.stringify(w)).NDCG10;

// =================== COVER ===================
const cover = [];
const cp = (t, o = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: o.before ?? 0, after: o.after ?? 120 },
  children: [new TextRun({ text: t, font: FONT, size: o.size ?? 26, bold: o.bold, italics: o.italics, color: o.color })] });
cover.push(cp("MANIPAL UNIVERSITY JAIPUR", { size: 34, bold: true, before: 400, color: "1F3864" }));
cover.push(cp("School of Computer Science and Engineering", { size: 26 }));
cover.push(cp("Department of Computer Science and Engineering", { size: 26, after: 700 }));
cover.push(cp("PROJECT REPORT", { size: 30, bold: true, after: 200 }));
cover.push(new Paragraph({ alignment: AlignmentType.CENTER, border: { top: { style: BorderStyle.SINGLE, size: 12, color: "1F3864", space: 8 }, bottom: { style: BorderStyle.SINGLE, size: 12, color: "1F3864", space: 8 } },
  spacing: { before: 100, after: 100 }, children: [new TextRun({ text: "Comparative Analysis of Existing Recommender Systems:", font: FONT, size: 38, bold: true }) ] }));
cover.push(cp("Neighbourhood, Latent-Factor, Content-Based and Hybrid Models Evaluated on the MovieTweetings Dataset", { size: 28, italics: true, before: 200, after: 800 }));
cover.push(cp("Submitted by", { size: 24, italics: true, after: 60 }));
cover.push(cp("Rhea", { size: 30, bold: true, after: 60 }));
cover.push(cp("Registration Number: [YOUR REGISTRATION NUMBER]", { size: 24, after: 60 }));
cover.push(cp("B.Tech (Computer Science and Engineering), Section C, Final Year", { size: 24, after: 400 }));
cover.push(cp("Submitted to", { size: 24, italics: true, after: 60 }));
cover.push(cp("[FACULTY / COURSE INSTRUCTOR NAME]", { size: 24, after: 60 }));
cover.push(cp("Course: [COURSE NAME AND CODE]", { size: 24, after: 500 }));
cover.push(cp("October 2026", { size: 24, bold: true }));

// =================== FRONT MATTER ===================
const body = [];
body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: "Abstract", font: FONT })] }));
body.push(P(`Recommender systems help users navigate very large catalogues by predicting which items they will value. A wide range of algorithms exists, but practitioners and students often struggle to know which family of methods to use, because accuracy on rating prediction and quality of top-N recommendation lists turn out to be very different things. This project presents a controlled, reproducible comparison of **eleven recommenders** on the public **MovieTweetings** dataset, after 20-core filtering **${nf(R.n_ratings)} ratings from ${nf(R.n_users)} users on ${nf(R.n_items)} movies**. The models cover non-personalised baselines (Random, Popularity, Bias baseline), memory-based collaborative filtering (User-KNN, Item-KNN), latent-factor models (truncated SVD, ALS matrix factorisation), implicit-feedback models (implicit Item-KNN, PureSVD), a genre-based content model, and a weighted hybrid. All algorithms were implemented from scratch with NumPy/SciPy; hyper-parameters were tuned on a separate validation split, and the final models were scored on a held-out 20% test split with RMSE and MAE for rating prediction and Precision, Recall, Hit-Rate and NDCG at 10 and 20 (full ranking over all unseen items), plus catalogue coverage and novelty.`));
body.push(P(`The results show a clear split. For **rating prediction**, Item-KNN is the most accurate (RMSE ${f3(F["Item-KNN"].RMSE)}, a ${gain}% reduction over the bias baseline), followed by ALS matrix factorisation (${f3(F["ALS Matrix Factorisation"].RMSE)}). For **top-N recommendation**, however, models trained on the binary interaction signal are roughly eight to ten times better: the best model, a hybrid of PureSVD, implicit Item-KNN and genre content, reaches NDCG@10 = ${f3(hyb["NDCG@10"])} and a hit-rate of ${pc(hyb["HR@10"], 0)}, against ${f3(pop["NDCG@10"])} for Popularity and only ${f3(F["Item-KNN"]["NDCG@10"])}–${f3(F["SVD (residual)"]["NDCG@10"])} for rating-optimised models. The hybrid's advantage over its two strongest components is statistically significant (paired t-test over ${nf(R.split.test_users)} users). We also analyse accuracy–diversity trade-offs, sensitivity to hyper-parameters, and performance across light, medium and heavy users. The main practical conclusion is that **the evaluation objective must match the product goal**: a model that minimises RMSE is generally not the model that produces the best recommendation list.`));
body.push(P(`**Keywords:** recommender systems, collaborative filtering, matrix factorisation, implicit feedback, hybrid recommendation, top-N evaluation, MovieTweetings.`, { before: 80 }));
body.push(pb());

body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: "Contents", font: FONT })] }));
["1. Introduction", "2. Literature Review", "3. Problem Formulation", "4. Dataset and Pre-processing", "5. Methodology and Model Design",
  "6. Experimental Setup and Evaluation Protocol", "7. Results and Discussion", "8. Limitations and Threats to Validity", "9. Conclusion and Future Work",
  "References", "Appendix A: Hyper-parameter Search Spaces and Selected Values", "Appendix B: Code Structure and Reproduction Steps"].forEach((t) =>
  body.push(new Paragraph({ spacing: { after: 90 }, indent: { left: 360 }, children: [new TextRun({ text: t, font: FONT, size: 24 })] })));
body.push(pb());

// =================== 1. INTRODUCTION ===================
body.push(H1("1. Introduction"));
body.push(H2("1.1 Background and Motivation"));
body.push(P(`Online platforms such as Netflix, YouTube, Amazon and Spotify present users with catalogues containing millions of items. Without assistance, users cannot discover relevant content, and platforms lose engagement. Recommender systems address this information-overload problem by learning from past behaviour and suggesting items a user is likely to enjoy. In industry, the quality of the recommender has a direct commercial impact; industrial descriptions of large video platforms show recommendation sitting at the centre of the product [17].`));
body.push(P(`Academic and industrial research has produced a very large number of algorithms, from the neighbourhood methods of the 1990s [1], [2] to latent-factor models popularised by the Netflix Prize [3] and, more recently, deep neural, graph and sequential models [9]–[13]. For a student or a small team that has to pick a method, this abundance creates two difficulties. First, results reported in individual papers are rarely comparable because datasets, splits and metrics differ. Second, several recent studies have shown that careful baselines are often much harder to beat than papers suggest [14], [18], [19]. A transparent, like-for-like comparison of well-understood methods therefore remains valuable.`));
body.push(H2("1.2 Problem Statement"));
body.push(P(`Given a log of user–movie ratings, build and compare a set of recommender systems and determine, under one common experimental protocol, (i) which method predicts held-out ratings most accurately, (ii) which method produces the best ranked list of unseen movies for each user, and (iii) how these methods differ in catalogue coverage, novelty and robustness to user activity level.`));
body.push(H2("1.3 Objectives"));
[`Review at least ten recent research papers on recommender systems and identify the research gap that motivates the project.`,
  `Formulate the recommendation problem formally, select a suitable public dataset and design a leakage-free train/validation/test protocol.`,
  `Implement representative models from each family (baselines, memory-based, latent-factor, implicit-feedback, content-based, hybrid) and tune them on validation data only.`,
  `Evaluate all models with both rating-prediction and top-N ranking metrics, and test whether differences are statistically significant.`,
  `Analyse accuracy–diversity trade-offs, hyper-parameter sensitivity and behaviour for light versus heavy users, and draw practical recommendations.`].forEach((t) => body.push(num(t)));
body.push(H2("1.4 Contributions"));
body.push(P(`The project (a) provides a clean, fully reproducible implementation of eleven recommenders in under 400 lines of NumPy/SciPy, (b) reports a side-by-side comparison under both rating-prediction and full-ranking evaluation, (c) quantifies how strongly the choice of objective changes the ranking of algorithms, and (d) documents a hybrid model whose component weights are chosen on validation data and whose advantage is checked with a paired significance test.`));

// =================== 2. LITERATURE REVIEW ===================
body.push(H1("2. Literature Review"));
body.push(P(`This section summarises ten recent studies (2016–2022) and seven foundational works that shaped the design of this project. The recent studies are grouped into three themes: neural and graph models, sequential/generative models, and critical re-evaluations of progress.`));
body.push(H2("2.1 Foundations of Collaborative and Content-Based Filtering"));
body.push(P(`Collaborative filtering (CF) predicts a user's preferences from the behaviour of similar users or items. GroupLens [1] introduced user-based neighbourhood CF for Usenet news, and Sarwar et al. [2] showed that item-based neighbourhoods are more stable and scalable because item–item similarities change slowly. Koren, Bell and Volinsky [3] described the latent-factor and bias models that dominated the Netflix Prize, and Zhou et al. [4] proposed Alternating Least Squares with weighted regularisation (ALS-WR), which we use for our matrix-factorisation model. Hu, Koren and Volinsky [5] showed how implicit feedback (clicks, views, purchases) can be modelled with confidence weights, and Cremonesi et al. [6] demonstrated that a plain truncated SVD on the binary interaction matrix (PureSVD) can beat far more elaborate rating-oriented models for top-N recommendation. Content-based filtering [7] and hybrid designs [8] complement CF when interaction data are sparse.`));
body.push(tcap("Table 1: Foundational works used in this project"));
body.push(table([2000, 2100, 2750, 2176], [
  ["Study", "Approach", "Key finding", "Use in this project"],
  ["Resnick et al., 1994 [1]", "User-based CF (GroupLens)", "Similar users' ratings predict a user's taste", "User-KNN"],
  ["Sarwar et al., 2001 [2]", "Item-based CF", "Item neighbourhoods are more scalable and stable", "Item-KNN"],
  ["Koren et al., 2009 [3]", "Biases + latent factors", "Latent-factor models outperformed neighbourhood models on Netflix data", "Bias baseline, SVD"],
  ["Zhou et al., 2008 [4]", "ALS-WR", "Regularisation scaled by number of ratings avoids over-fitting", "ALS matrix factorisation"],
  ["Hu et al., 2008 [5]", "Implicit-feedback MF", "Treat interactions as positive signal with confidence", "Binary-signal models"],
  ["Cremonesi et al., 2010 [6]", "PureSVD", "Simple SVD on binary data wins top-N tasks", "PureSVD"],
  ["Lops et al., 2011 [7]; Burke, 2002 [8]", "Content-based; hybrids", "Combining signals reduces sparsity and cold-start problems", "Content model, Hybrid"],
], { left: [0, 1, 2, 3], size: 18 }));
body.push(H2("2.2 Recent Studies (2016–2022)"));
body.push(P(`**Neural and graph models.** He et al. [9] proposed Neural Collaborative Filtering (NCF), replacing the inner product with a multi-layer perceptron. LightGCN [10] simplified graph convolution networks for CF by removing feature transformations and non-linearities and obtained strong results on benchmark data. Wu et al. [16] survey the growing family of graph neural recommenders, and Zhang et al. [15] survey deep-learning-based recommenders in general. Covington et al. [17] describe a two-stage deep candidate-generation and ranking system used at YouTube scale.`));
body.push(P(`**Sequential and generative models.** SASRec [11] applies self-attention to a user's item sequence to predict the next item, and BERT4Rec [12] uses bidirectional transformers trained with a masked-item objective. Mult-VAE [13] applies variational autoencoders with a multinomial likelihood to implicit-feedback CF.`));
body.push(P(`**Critical re-evaluation.** Ferrari Dacrema et al. [14] attempted to reproduce neural recommendation methods published at top venues and found that most could not be reproduced or were outperformed by comparatively simple, well-tuned baselines such as nearest-neighbour methods; the extended study appeared in ACM TOIS [18]. Rendle et al. [19] revisited NCF and showed that a properly tuned dot-product matrix-factorisation model beats the MLP-based similarity. Krichene and Rendle [20] showed that evaluating on sampled negative items can change model rankings, which is why this project ranks against all unseen items.`));
body.push(tcap("Table 2: Summary of ten recent studies (2016–2022)"));
body.push(table([1750, 1600, 3000, 2676], [
  ["Study", "Approach", "Main idea / finding", "Relevance to this project"],
  ["Covington et al., 2016 [17]", "Deep candidate generation + ranking", "Industrial two-stage neural recommender at YouTube scale", "Motivates candidate-ranking view of top-N"],
  ["He et al., 2017 [9]", "Neural CF (MLP)", "Learns the user–item interaction function with an MLP", "Future work; compared against in [19]"],
  ["Liang et al., 2018 [13]", "Mult-VAE", "Variational autoencoder with multinomial likelihood for implicit data", "Supports using implicit signal"],
  ["Kang & McAuley, 2018 [11]", "SASRec (self-attention)", "Models item order to predict next interaction", "Future work (sequential, temporal split)"],
  ["Sun et al., 2019 [12]", "BERT4Rec", "Bidirectional transformer with masked-item training", "Future work"],
  ["Zhang et al., 2019 [15]", "Survey", "Taxonomy of deep-learning recommenders and open issues", "Positions our classical baselines"],
  ["Ferrari Dacrema et al., 2019 [14]", "Reproducibility study", "Most neural methods not reproducible or beaten by simple baselines", "Justifies strong tuned baselines"],
  ["Rendle et al., 2020 [19]", "NCF vs MF revisited", "Tuned dot-product MF beats MLP similarity", "Supports MF/PureSVD choice"],
  ["He et al., 2020 [10]", "LightGCN", "Simplified graph convolution for CF", "Future work (graph models)"],
  ["Wu et al., 2022 [16]", "GNN survey", "Reviews graph neural network recommenders", "Context for future work"],
], { left: [0, 1, 2, 3], size: 18 }));
body.push(H2("2.3 Research Gap and Positioning"));
body.push(P(`The literature shows two consistent messages: modern neural architectures promise gains, yet well-tuned classical methods remain highly competitive, and the choice of evaluation protocol can change conclusions [14], [19], [20]. What is often missing in student and applied settings is a compact side-by-side study that (i) covers all major classical families, (ii) uses the same data split and tuning budget for each, (iii) reports **both** rating-prediction and top-N metrics, and (iv) tests significance. This project fills that gap and uses it as a baseline against which future neural models can be compared.`));

// =================== 3. PROBLEM FORMULATION ===================
body.push(H1("3. Problem Formulation"));
body.push(P(`Let U be the set of users, I the set of items and R ⊂ U × I the set of observed interactions. For each (u, i) in R we observe a rating r_ui on a 0–10 scale. The data are very sparse: only a small fraction of the |U| × |I| matrix is observed. We study two tasks.`));
body.push(P(`**Task A – Rating prediction.** Learn a function r̂(u, i) that minimises the error on held-out ratings:`));
body.push(eq("RMSE = √( (1/|T|) Σ_(u,i)∈T ( r̂_ui − r_ui )² ),     MAE = (1/|T|) Σ_(u,i)∈T | r̂_ui − r_ui |"));
body.push(P(`**Task B – Top-N recommendation.** For every user u, produce a ranked list of N items from I \\ I_u (items not yet rated in training) such that the items the user later likes appear near the top. An item in the held-out set is defined as *relevant* if the user rated it **≥ 8 out of 10**. The ranking is computed over **all** unseen items (roughly 4,700 candidate movies per user), not over a sampled subset of negatives.`));
body.push(P(`A model can be good at Task A and poor at Task B and vice versa, because Task A scores only items that users chose to rate, while Task B asks the model to choose among the whole catalogue. The comparison of these two views is the central theme of this report.`));

// =================== 4. DATASET ===================
body.push(H1("4. Dataset and Pre-processing"));
body.push(H2("4.1 Dataset Selection"));
body.push(P(`We use **MovieTweetings** [21], a public dataset of movie ratings extracted from well-structured tweets generated by the IMDb mobile apps (ratings are on a 0–10 scale) together with movie titles and genres. It was chosen because it (a) is freely available and widely used in recommender-systems research, (b) contains a larger number of users and items than the classic MovieLens-100K benchmark [22] while remaining small enough to run all experiments on a single CPU core in a few minutes, (c) provides genre metadata required for the content-based and hybrid models, and (d) covers recent films, which makes the qualitative examples easier to interpret.`));
body.push(H2("4.2 Pre-processing"));
body.push(P(`The raw snapshot contains ${nf(R.raw.n_ratings)} ratings from ${nf(R.raw.n_users)} users on ${nf(R.raw.n_items)} movies. Because users and items with only a handful of ratings carry too little signal for reliable evaluation, we applied iterative **20-core filtering**: users and items with fewer than 20 ratings were removed repeatedly until every remaining user and item had at least 20 ratings. Duplicates were dropped, and user and item IDs were re-indexed to contiguous integers. Titles were HTML-unescaped, and movie release years were extracted from titles for a decade feature. Table 3 gives the resulting statistics.`));
body.push(tcap("Table 3: Dataset statistics before and after 20-core filtering"));
body.push(table([4200, 2413, 2413], [
  ["Statistic", "Raw", "After filtering"],
  ["Users", nf(R.raw.n_users), nf(R.n_users)],
  ["Items (movies)", nf(R.raw.n_items), nf(R.n_items)],
  ["Ratings", nf(R.raw.n_ratings), nf(R.n_ratings)],
  ["Density", "—", pc(R.density, 2)],
  ["Mean rating (0–10)", "—", R.mean_rating.toFixed(2)],
  ["Median ratings per user", "—", R.median_user_cnt.toFixed(0)],
  ["Median ratings per item", "—", R.median_item_cnt.toFixed(0)],
  ["Most frequent genres", "—", Object.entries(R.top_genres).slice(0, 4).map(([g, c]) => `${g}`).join(", ")],
], { left: [0] }));
body.push(H2("4.3 Exploratory Analysis"));
body.push(fig("eda", 600));
body.push(cap("Figure 1: Rating distribution (left) and long-tail distribution of activity per user (centre) and popularity per item (right), log–log axes."));
body.push(P(`Three properties of the data shape the later results. (i) Ratings are skewed to the high side (mean ${R.mean_rating.toFixed(2)}; the modes are 7–9), so the bias terms carry much of the explainable variance and a high threshold (≥ 8) is needed to define a meaningful "like". (ii) Both user activity and item popularity follow a **long-tail** (power-law-like) distribution: a few blockbusters receive a very large share of ratings. This makes a popularity recommender a surprisingly strong baseline and makes popularity bias an important evaluation issue [23], [24]. (iii) The matrix is ${pc(R.density, 2)} dense, which favours methods that regularise heavily or use neighbourhood shrinkage.`));
body.push(H2("4.4 Data Splitting Strategy"));
body.push(P(`To prevent leakage we use a **per-user random split**. First, 20% of each user's ratings are held out as the **test set** (${nf(R.split.test)} ratings). From the remaining 80%, 12.5% of each user's ratings (10% of all data) form a **validation set** (${nf(R.split.val)} ratings) used exclusively for hyper-parameter and hybrid-weight selection, with the remaining ${nf(R.split.train)} ratings as the tuning-time training set. After tuning, every model is re-trained on the full 80% training portion (${nf(R.split.train + R.split.val)} ratings) and evaluated once on the test set. Users with no relevant (≥ 8) held-out rating cannot be scored on ranking metrics; ${nf(R.split.test_users)} test users remain. The random seed is fixed (42) for reproducibility.`));

// =================== 5. METHODOLOGY ===================
body.push(H1("5. Methodology and Model Design"));
body.push(P(`Eleven models were implemented behind a common interface (fit on a training split; return a score for every item for a batch of users). Table 4 gives an overview; the following subsections describe each.`));
body.push(tcap("Table 4: Overview of the evaluated recommenders"));
body.push(table([2300, 1700, 1700, 3326], [
  ["Model", "Family", "Input signal", "Score used for ranking"],
  ["Random", "Baseline", "None", "Uniform random score"],
  ["Popularity", "Baseline", "Interaction counts", "Number of training ratings of the item"],
  ["Bias baseline", "Baseline / rating", "Ratings", "μ + b_u + b_i"],
  ["User-KNN", "Memory-based CF", "Rating residuals", "Predicted rating from k similar users"],
  ["Item-KNN", "Memory-based CF", "Rating residuals", "Predicted rating from k similar items"],
  ["SVD (residual)", "Latent factor", "Rating residuals", "Predicted rating from truncated SVD"],
  ["ALS MF", "Latent factor", "Rating residuals", "Predicted rating from ALS factors"],
  ["Item-KNN (implicit)", "Implicit CF", "Binary interactions", "Sum of similarities to the user's items"],
  ["PureSVD (binary)", "Implicit latent factor", "Binary interactions", "Rank-k reconstruction of the user row"],
  ["Content-based", "Content", "Genres, decade, ratings", "Cosine(user profile, item vector)"],
  ["Hybrid", "Weighted hybrid", "All of the above", "Weighted sum of z-scored component scores"],
], { left: [0, 1, 2, 3], size: 18 }));
body.push(H2("5.1 Baselines"));
body.push(P(`**Random** and **Popularity** are non-personalised reference points. The **bias baseline** predicts a rating as the global mean plus user and item offsets, with damped (shrunken) estimates to avoid over-fitting for rarely rated users and items [3], [26]:`));
body.push(eq("r̂_ui = μ + b_u + b_i"));
body.push(eq("b_i = Σ_u (r_ui − μ) / (n_i + λ_i),     b_u = Σ_i (r_ui − μ − b_i) / (n_u + λ_u)"));
body.push(P(`with λ_i = λ_u = 10 and n_i, n_u the number of ratings of item i and user u. All rating-based models below predict the **residual** r̃_ui = r_ui − (μ + b_u + b_i) and add the baseline back at prediction time, a standard trick that removes the strongest user and item effects.`));
body.push(H2("5.2 Memory-Based Collaborative Filtering"));
body.push(P(`**User-KNN** and **Item-KNN** compute cosine similarity between rows (users) or columns (items) of the residual matrix. To prevent high similarities based on very few co-rated entries, similarities are shrunk by the co-rating support c_uv [26]:`));
body.push(eq("sim(u, v) = cos( r̃_u , r̃_v ) · c_uv / ( c_uv + λ_s ),     λ_s = 20"));
body.push(P(`Only the k most similar neighbours with positive similarity are kept. The prediction for item i is the baseline plus a similarity-weighted average of neighbours' residuals, with a small damping term δ in the denominator so that predictions based on little evidence fall back to the baseline:`));
body.push(eq("r̂_ui = b_ui + [ Σ_(v∈N(u)) sim(u,v) · r̃_vi ] / [ Σ_(v∈N(u)) |sim(u,v)| · 1[v rated i] + δ ]"));
body.push(P(`Item-KNN is symmetric, with neighbours taken over items the user has rated. The neighbourhood size k and δ are tuned on validation data.`));
body.push(H2("5.3 Latent-Factor Models"));
body.push(P(`**SVD (residual)** factorises the sparse residual matrix with a truncated singular value decomposition, R̃ ≈ U_k Σ_k V_kᵀ, and uses the reconstruction as the residual prediction (missing entries are treated as zero residual).`));
body.push(P(`**ALS Matrix Factorisation** learns user and item factor vectors p_u, q_i ∈ ℝ^f by minimising the regularised squared error over observed entries only, with regularisation proportional to the number of ratings of each user and item (ALS-WR [4]):`));
body.push(eq("min Σ_(u,i)∈R ( r̃_ui − p_u·q_i )² + λ ( Σ_u n_u ‖p_u‖² + Σ_i n_i ‖q_i‖² )"));
body.push(P(`With the item factors fixed, the optimal p_u is the closed-form solution of a small ridge-regression problem (and vice versa), so training alternates between solving all user vectors and all item vectors for ${hist.length} iterations. The number of factors f and λ are tuned on validation data.`));
body.push(H2("5.4 Implicit-Feedback Models"));
body.push(P(`Whether a user *chose* to rate a movie carries information beyond the rating value. We therefore built two models on the binary matrix B (1 if the user rated the item in training). **Item-KNN (implicit)** computes shrunk cosine similarity between binary item columns and scores an unseen item by the sum of its similarities to the items already in the user's history: score(u, i) = Σ_{j∈I_u ∩ N_k(i)} sim(i, j). **PureSVD** [6] computes a truncated SVD of B, B ≈ U_k Σ_k V_kᵀ, and scores every item by the reconstructed entry.`));
body.push(H2("5.5 Content-Based Model"));
body.push(P(`Each movie is represented by a vector of genre indicators and a release-decade indicator, weighted by inverse document frequency (TF-IDF) and L2-normalised. A user profile is the sum of the vectors of the movies the user rated, weighted by the rating minus the user's mean rating (so disliked genres contribute negatively), again normalised. Items are ranked by cosine similarity to the profile. This model needs no other users' data, so it is useful for new items, but genre alone is a coarse description.`));
body.push(H2("5.6 Hybrid Model"));
body.push(P(`The hybrid combines PureSVD, Item-KNN (implicit) and the content model. Component scores live on different scales, so for each user the scores of unseen items are converted to z-scores and then linearly combined:`));
body.push(eq("score_hyb(u, i) = w₁ · z(PureSVD) + w₂ · z(Item-KNN impl.) + w₃ · z(Content)"));
body.push(P(`Weights (w₁, w₂, w₃) were chosen from a 12-point grid by validation NDCG@10 (Section 7.6 and Appendix A).`));
body.push(H2("5.7 Training and Tuning Procedure"));
body.push(P(`All hyper-parameters were selected using the validation split only (on a fixed sample of 3,000 validation users to keep each trial short): neighbourhood size k ∈ {5, 10, 20, 50, 100, 200}, damping δ ∈ {0.05, 0.2, 0.5, 1, 2}, SVD rank, PureSVD rank, ALS factors and regularisation (3 × 3 grid) and hybrid weights. Rating-based models were selected by validation RMSE; ranking-only models by validation NDCG@10. The selected values are listed in Appendix A. The test set was used once, for the final report.`));

// =================== 6. EXPERIMENTAL SETUP ===================
body.push(H1("6. Experimental Setup and Evaluation Protocol"));
body.push(H2("6.1 Environment"));
body.push(P(`All experiments ran on a single CPU core with about 4 GB of RAM using Python 3.12, NumPy, SciPy (sparse matrices and svds), pandas and Matplotlib. No GPU or deep-learning library was needed. The complete pipeline (tuning, final training of all eleven models and test evaluation) takes roughly five minutes.`));
body.push(H2("6.2 Evaluation Metrics"));
body.push(P(`**Rating prediction:** RMSE and MAE on all held-out ratings (predictions clipped to [0, 10]). **Top-N ranking** at K ∈ {10, 20}, over all items the user has not rated in training, with relevant = held-out rating ≥ 8:`));
body.push(bullet(`**Precision@K** – fraction of the K recommended items that are relevant; **Recall@K** – fraction of the user's relevant items retrieved; **Hit-Rate@K** – fraction of users with at least one relevant item in the list.`));
body.push(bullet(`**NDCG@K** – normalised discounted cumulative gain [25], NDCG = DCG / IDCG with DCG = Σ_(r=1..K) rel_r / log₂(r + 1); it rewards placing relevant items near the top. NDCG@10 is our primary metric.`));
body.push(bullet(`**Catalogue coverage@10** – share of all movies that appear in at least one user's top-10; **Novelty@10** – mean of −log₂(popularity share) of recommended items (higher = less mainstream).`));
body.push(P(`Per-user NDCG@10 values are retained so that 95% confidence intervals (normal approximation) and paired t-tests between models can be computed on the same users.`));

// =================== 7. RESULTS ===================
body.push(H1("7. Results and Discussion"));
body.push(H2("7.1 Rating Prediction Accuracy"));
body.push(tcap("Table 5: Rating-prediction error on the held-out test set (lower is better)"));
const rrows = [["Model", "RMSE", "MAE", "Improvement in RMSE vs. bias baseline"]];
ratingModels.forEach((m) => rrows.push([m, f4(F[m].RMSE), f4(F[m].MAE), m === "Bias baseline" ? "—" : ((F["Bias baseline"].RMSE - F[m].RMSE) / F["Bias baseline"].RMSE * 100).toFixed(2) + "%"]));
body.push(table([3000, 1500, 1500, 3026], rrows, { hl: [rrows.findIndex((r) => r[0] === bestRating)] }));
body.push(fig("rating_accuracy", 560));
body.push(cap("Figure 2: RMSE and MAE of all rating-prediction models on the test set."));
body.push(P(`**${bestRating}** is the most accurate rating predictor (RMSE ${f4(F[bestRating].RMSE)}), a ${gain}% improvement over the bias baseline. ALS matrix factorisation (${f4(F["ALS Matrix Factorisation"].RMSE)}) and User-KNN (${f4(F["User-KNN"].RMSE)}) are close behind. The gains over the baseline are modest in absolute terms, which is typical: on a 0–10 scale with strongly skewed ratings, most of the predictable signal is already captured by the user and item biases. The ordering Item-KNN > ALS > User-KNN > SVD is consistent with the literature that item-based neighbourhoods are accurate and stable [2]. Plain truncated SVD on residuals with zeros for missing entries gives the smallest gain because it treats unobserved entries as observed zero residuals; ALS, which fits only observed entries, avoids this.`));
body.push(H2("7.2 Top-N Recommendation Quality"));
body.push(tcap("Table 6: Top-N ranking metrics on the test set (all unseen items ranked; relevant = rating ≥ 8)"));
const krows = [["Model", "P@10", "R@10", "HR@10", "NDCG@10", "NDCG@20"]];
order.forEach((m) => krows.push([m.replace("Hybrid (PureSVD+KNN+Content)", "Hybrid (best)"), f3(F[m]["P@10"]), f3(F[m]["R@10"]), pc(F[m]["HR@10"]), f3(F[m]["NDCG@10"]) + " ± " + f3(F[m]["NDCG@10_ci"]), f3(F[m]["NDCG@20"])]));
body.push(table([2800, 1000, 1000, 1100, 1900, 1226], krows, { hl: [krows.length - 1], size: 18 }));
body.push(P(`± denotes the 95% confidence interval of mean NDCG@10 over ${nf(R.split.test_users)} users.`, { size: 19, before: 60 }));
body.push(fig("ranking", 610));
body.push(cap("Figure 3: Top-10 ranking quality of the eleven models."));
body.push(P(`The ranking results are strikingly different from the rating results. The three models trained on the **binary interaction signal** – the hybrid (NDCG@10 ${f3(hyb["NDCG@10"])}), PureSVD (${f3(psv["NDCG@10"])}) and implicit Item-KNN (${f3(iki["NDCG@10"])}) – are far ahead of everything else. The hybrid retrieves at least one liked movie in the top-10 for ${pc(hyb["HR@10"], 0)} of users and ${pc(hyb["R@10"], 0)} of each user's liked held-out movies on average. The simple **Popularity** baseline (NDCG@10 ${f3(pop["NDCG@10"])}) is ${(pop["NDCG@10"] / F["Item-KNN"]["NDCG@10"]).toFixed(1)} times better than the most accurate rating predictor, Item-KNN (${f3(F["Item-KNN"]["NDCG@10"])}).`));
body.push(P(`This reversal agrees with earlier findings [6], [23] and has a clear explanation. Models that rank by *predicted rating* favour items with few ratings whose small, noisy evidence happens to be positive; those items are rarely the ones a user actually goes on to watch and rate. Task A is evaluated only on items that were rated, so it never penalises this behaviour, whereas Task B asks the model to pick the right items from the whole catalogue. The binary models instead learn *which items tend to be rated together*, which is the dominant signal for what a user will interact with next. In practice this means that a team who optimises only RMSE may deploy a model that is no better than random-plus-bias for their actual ranked-list product.`));
body.push(H2("7.3 Statistical Significance"));
const tp = R.paired_t_vs_best;
body.push(P(`Paired t-tests on per-user NDCG@10 show that the hybrid's advantage over PureSVD (t = ${tp["PureSVD (binary)"].toFixed(1)}) and implicit Item-KNN (t = ${tp["Item-KNN (implicit)"].toFixed(1)}) is highly significant (p < 0.001; the critical value at 95% is about 1.96). The difference between the top models is small in absolute size (${f3(hyb["NDCG@10"] - psv["NDCG@10"])} NDCG points over PureSVD, a ${((hyb["NDCG@10"] / psv["NDCG@10"] - 1) * 100).toFixed(1)}% relative gain), but it is consistent across users. All gaps between the binary-signal models and the rating-based models are far larger than the confidence intervals in Table 6.`));
body.push(H2("7.4 Accuracy versus Diversity and Novelty"));
body.push(tcap("Table 7: Coverage, novelty and popularity of the top-10 lists"));
const drows = [["Model", "NDCG@10", "Coverage@10", "Novelty@10 (bits)", "Avg. #ratings of recommended items"]];
order.forEach((m) => drows.push([m.replace("Hybrid (PureSVD+KNN+Content)", "Hybrid (best)"), f3(F[m]["NDCG@10"]), pc(F[m]["Coverage@10"]), F[m]["Novelty@10"].toFixed(2), nf(Math.round(F[m]["AvgPop@10"]))]));
body.push(table([2700, 1200, 1500, 1700, 1926], drows, { hl: [drows.length - 1], size: 18 }));
body.push(fig("tradeoff", 560));
body.push(cap("Figure 4: Accuracy–coverage (left) and accuracy–novelty (right) trade-offs."));
body.push(P(`Accuracy is bought with popularity. Popularity recommends the same few titles to everybody (coverage ${pc(pop["Coverage@10"])}, average item has ${nf(Math.round(pop["AvgPop@10"]))} ratings), while the best-performing hybrid still recommends relatively mainstream movies (${nf(Math.round(hyb["AvgPop@10"]))} ratings on average) but touches ${pc(hyb["Coverage@10"])} of the catalogue, more than ${(hyb["Coverage@10"] / pop["Coverage@10"]).toFixed(0)} times the coverage of Popularity. Random and the genre-based model reach 97–100% coverage and high novelty but have almost no accuracy. The hybrid therefore gives the best accuracy while keeping personalised, non-trivial lists, yet it does not solve the popularity bias: a diversity-aware re-ranking step would be a natural extension.`));
body.push(H2("7.5 Behaviour by User Activity"));
body.push(tcap("Table 8: NDCG@10 by user activity tercile (number of training ratings)"));
const segRows = [["Model", `Light (≤ ${seg.cuts[0]})`, `Medium (≤ ${seg.cuts[1]})`, `Heavy (> ${seg.cuts[1]})`]];
Object.keys(sn).forEach((m) => segRows.push([m.replace("Hybrid (PureSVD+KNN+Content)", "Hybrid (best)"), f3(sn[m].Light), f3(sn[m].Medium), f3(sn[m].Heavy)]));
body.push(table([3500, 1842, 1842, 1842], segRows, { hl: [segRows.length - 1] }));
body.push(fig("segments", 430));
body.push(cap("Figure 5: NDCG@10 for light, medium and heavy users."));
body.push(P(`Every method scores highest for heavy users, whose longer histories give more evidence. In relative terms, however, personalisation helps *most* where history is short: the hybrid is ${lift("Light")} times better than Popularity for light users, ${lift("Medium")} times for medium users and ${lift("Heavy")} times for heavy users, because heavy users' tastes overlap strongly with the mainstream that Popularity already captures. We note that our 20-core filter keeps at least 16 training ratings per user, so this is a "light" rather than a true cold-start analysis; genuine cold-start users would need the content model or onboarding questions.`));
body.push(H2("7.6 Hyper-parameter Sensitivity and Hybrid Ablation"));
body.push(fig("sensitivity", 610));
body.push(cap("Figure 6: Validation RMSE versus neighbourhood size (left) and number of latent factors (centre); ALS convergence on train and test RMSE (right)."));
body.push(P(`**Neighbourhood size.** Validation RMSE falls steadily as k grows from 5 to 200 for both KNN models and Item-KNN is better than User-KNN at every k; the best value lies at the edge of the grid (k = ${B.item_k}), so a larger k might help slightly. **Latent factors.** Un-regularised or weakly regularised ALS over-fits badly (validation RMSE worse than the bias baseline for λ = 0.15 with few factors), whereas with λ = ${B.als_lam} and f = ${B.als_k} factors ALS becomes the second-best rating model; this confirms the importance of the weighted regularisation in [4]. **Convergence.** ALS training RMSE keeps decreasing (to ${hl[1].toFixed(2)} after ${hl[0]} iterations) while test RMSE flattens near ${hl[2].toFixed(2)} after about eight iterations, so the large train–test gap reflects the sparsity of the data rather than lack of training.`));
body.push(P(`**Hybrid ablation.** On validation data, a 50/50 blend of PureSVD and implicit Item-KNN already reaches NDCG@10 = ${f3(hvGet([0.5, 0.5, 0]))}, higher than either alone (${f3(hvGet([1, 0, 0]))} and ${f3(hvGet([0, 1, 0]))}); adding 20% weight on the genre model raises it to ${f3(hvGet([0.4, 0.4, 0.2]))}. A pure content model scores only ${f3(hvGet([0, 0, 1]))}. Hence most of the hybrid's gain comes from **ensembling two different collaborative signals**, with a smaller contribution from content, which is nevertheless the only component that works for items with no interaction history.`));
body.push(H2("7.7 Qualitative Example"));
const demo = R.demo; const dr = demo.recs;
body.push(P(`To sanity-check the lists, we inspected one randomly chosen user (re-indexed ID ${demo.user}). Their highest-rated training movies include ${demo.liked.slice(0, 5).map((x) => x[0]).join(", ")}. Table 9 shows the top-5 recommendations of selected models.`));
body.push(tcap("Table 9: Top-5 recommendations for the example user"));
const names2 = ["Popularity", "Content-based (genres)", "Item-KNN", "PureSVD (binary)", "Hybrid (PureSVD+KNN+Content)"];
const qrows = [["Rank", ...names2.map((n) => n.replace("Hybrid (PureSVD+KNN+Content)", "Hybrid").replace(" (genres)", "").replace(" (binary)", ""))]];
for (let i = 0; i < 5; i++) qrows.push([String(i + 1), ...names2.map((n) => dr[n][i])]);
body.push(table([600, 1600, 1700, 1700, 1700, 1726], qrows, { size: 16 }));
body.push(P(`The Popularity list is identical for every user. The rating-based Item-KNN recommends acclaimed classics (e.g., 12 Angry Men, The Godfather Part II), which are plausible but older than the recent releases this user rates. The binary-signal models (PureSVD and Hybrid) recommend recent films (2013–2014 releases) that match the era and style of the user's history; this user's later liked movies (e.g., ${demo.held_out_liked.slice(0, 4).join(", ")}) come from the same period, even though none of these five exact titles was among them. This illustrates why single examples are poor evidence, and why we rely on aggregate metrics.`, { before: 100 }));
body.push(H2("7.8 Key Findings"));
[`**Objective matters more than algorithm family.** The ranking of methods flips between rating prediction and top-N recommendation. Item-KNN wins on RMSE (${f3(F["Item-KNN"].RMSE)}) but is ${(pop["NDCG@10"] / F["Item-KNN"]["NDCG@10"]).toFixed(1)}× worse than plain Popularity on NDCG@10.`,
  `**Simple implicit models are very strong.** PureSVD (a truncated SVD on binary data) and implicit Item-KNN, with no learning loop at all, give NDCG@10 of ${f3(psv["NDCG@10"])} and ${f3(iki["NDCG@10"])}, consistent with [6], [14], [19].`,
  `**Hybridisation helps modestly but significantly.** The weighted hybrid is best on every ranking metric (NDCG@10 ${f3(hyb["NDCG@10"])}, Recall@10 ${f3(hyb["R@10"])}), with paired t-statistics above 9 versus its two strongest components.`,
  `**Regularisation is essential** for matrix factorisation on sparse data; weak regularisation is worse than the bias baseline.`,
  `**Popularity bias remains.** The best lists still concentrate on popular movies (about ${nf(Math.round(hyb["AvgPop@10"]))} training ratings per recommended item vs. a median of ${R.median_item_cnt.toFixed(0)} ratings per item in the full dataset), and coverage is ${pc(hyb["Coverage@10"], 0)}.`].forEach((t) => body.push(num(t)));

// =================== 8. LIMITATIONS ===================
body.push(H1("8. Limitations and Threats to Validity"));
[`**Random per-user split.** A temporal split would better mimic deployment and avoids using future data to predict the past; the strong results of the binary models may partly reflect that a random split lets the model see the same period's popularity trends.`,
  `**Single dataset and single seed.** All numbers come from one dataset and one split. Confidence intervals reflect user sampling variance, not split-to-split variance; replication on MovieLens-1M or a different domain is desirable.`,
  `**Implicit relevance from explicit ratings.** "Relevant" was defined as rating ≥ 8, which is a modelling choice. Self-selection (users tweet ratings for films they chose to watch and share) introduces bias.`,
  `**20-core filtering.** Users and items with fewer than 20 ratings are excluded, so true cold-start behaviour is not measured.`,
  `**Limited hyper-parameter budget.** Grids are small and tuned on 3,000 validation users; some optima lie at grid edges (k = ${B.item_k}).`,
  `**No deep models.** Neural, graph and sequential models were reviewed but not implemented; the report is therefore a baseline study, not a state-of-the-art claim.`].forEach((t) => body.push(bullet(t)));

// =================== 9. CONCLUSION ===================
body.push(H1("9. Conclusion and Future Work"));
body.push(P(`This project compared eleven recommender systems on the MovieTweetings dataset under a single leakage-free protocol. For rating prediction, Item-KNN with shrinkage and damping (RMSE ${f3(F["Item-KNN"].RMSE)}) and ALS matrix factorisation (${f3(F["ALS Matrix Factorisation"].RMSE)}) were the best, but improved on the bias baseline by only about ${gain}%. For top-N recommendation, models that exploit the binary interaction signal were roughly eight to ten times better than rating-optimised models, and a hybrid of PureSVD, implicit Item-KNN and content features achieved the best results (NDCG@10 ${f3(hyb["NDCG@10"])}, hit-rate ${pc(hyb["HR@10"], 0)}), significantly ahead of every other model. The study also highlighted the popularity bias that persists even in the best lists and showed that personalisation brings the largest relative benefit to users with short histories.`));
body.push(P(`The practical lesson is to **choose the evaluation metric that matches the product**: if the product is a ranked list, optimise and report ranking metrics over the whole catalogue, and always include a Popularity baseline.`));
body.push(P(`**Future work** includes: (1) temporal (leave-last-out) evaluation and replication on MovieLens-1M; (2) neural models such as NCF [9], Mult-VAE [13], LightGCN [10] and sequential models SASRec [11] and BERT4Rec [12], using the present results as the baseline set; (3) richer content features (plot text embeddings, cast/director) for the content and hybrid components; (4) diversity- and novelty-aware re-ranking and fairness analysis of popularity bias; (5) a learned (rather than grid-searched) hybrid combination; and (6) a small interactive demo for the best model.`));

// =================== REFERENCES ===================
body.push(pb());
body.push(H1("References"));
const refs = [
  `P. Resnick, N. Iacovou, M. Suchak, P. Bergstrom and J. Riedl, "GroupLens: An open architecture for collaborative filtering of netnews," in *Proc. ACM CSCW*, 1994.`,
  `B. Sarwar, G. Karypis, J. Konstan and J. Riedl, "Item-based collaborative filtering recommendation algorithms," in *Proc. 10th International World Wide Web Conference (WWW)*, 2001.`,
  `Y. Koren, R. Bell and C. Volinsky, "Matrix factorization techniques for recommender systems," *IEEE Computer*, vol. 42, no. 8, 2009.`,
  `Y. Zhou, D. Wilkinson, R. Schreiber and R. Pan, "Large-scale parallel collaborative filtering for the Netflix Prize," in *Proc. Algorithmic Aspects in Information and Management (AAIM)*, 2008.`,
  `Y. Hu, Y. Koren and C. Volinsky, "Collaborative filtering for implicit feedback datasets," in *Proc. IEEE ICDM*, 2008.`,
  `P. Cremonesi, Y. Koren and R. Turrin, "Performance of recommender algorithms on top-N recommendation tasks," in *Proc. ACM RecSys*, 2010.`,
  `P. Lops, M. de Gemmis and G. Semeraro, "Content-based recommender systems: State of the art and trends," in *Recommender Systems Handbook*, Springer, 2011.`,
  `R. Burke, "Hybrid recommender systems: Survey and experiments," *User Modeling and User-Adapted Interaction*, vol. 12, no. 4, 2002.`,
  `X. He, L. Liao, H. Zhang, L. Nie, X. Hu and T.-S. Chua, "Neural collaborative filtering," in *Proc. WWW*, 2017.`,
  `X. He, K. Deng, X. Wang, Y. Li, Y. Zhang and M. Wang, "LightGCN: Simplifying and powering graph convolution network for recommendation," in *Proc. ACM SIGIR*, 2020.`,
  `W.-C. Kang and J. McAuley, "Self-attentive sequential recommendation," in *Proc. IEEE ICDM*, 2018.`,
  `F. Sun et al., "BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer," in *Proc. ACM CIKM*, 2019.`,
  `D. Liang, R. G. Krishnan, M. D. Hoffman and T. Jebara, "Variational autoencoders for collaborative filtering," in *Proc. WWW*, 2018.`,
  `M. Ferrari Dacrema, P. Cremonesi and D. Jannach, "Are we really making much progress? A worrying analysis of recent neural recommendation approaches," in *Proc. ACM RecSys*, 2019.`,
  `S. Zhang, L. Yao, A. Sun and Y. Tay, "Deep learning based recommender system: A survey and new perspectives," *ACM Computing Surveys*, vol. 52, no. 1, 2019.`,
  `S. Wu, F. Sun, W. Zhang, X. Xie and B. Cui, "Graph neural networks in recommender systems: A survey," *ACM Computing Surveys*, vol. 55, no. 5, 2022.`,
  `P. Covington, J. Adams and E. Sargin, "Deep neural networks for YouTube recommendations," in *Proc. ACM RecSys*, 2016.`,
  `M. Ferrari Dacrema, S. Boglio, P. Cremonesi and D. Jannach, "A troubling analysis of reproducibility and progress in recommender systems research," *ACM Transactions on Information Systems*, vol. 39, no. 2, 2021.`,
  `S. Rendle, W. Krichene, L. Zhang and J. Anderson, "Neural collaborative filtering vs. matrix factorization revisited," in *Proc. ACM RecSys*, 2020.`,
  `W. Krichene and S. Rendle, "On sampled metrics for item recommendation," in *Proc. ACM SIGKDD*, 2020.`,
  `S. Dooms, T. De Pessemier and L. Martens, "MovieTweetings: A movie rating dataset collected from Twitter," in *Workshop on Crowdsourcing and Human Computation for Recommender Systems (CrowdRec) at ACM RecSys*, 2013.`,
  `F. M. Harper and J. A. Konstan, "The MovieLens datasets: History and context," *ACM Transactions on Interactive Intelligent Systems*, vol. 5, no. 4, 2015.`,
  `H. Steck, "Item popularity and recommendation accuracy," in *Proc. ACM RecSys*, 2011.`,
  `H. Abdollahpouri, M. Mansoury, R. Burke and B. Mobasher, "The unfairness of popularity bias in recommendation," in *Workshop on Recommendation in Multi-stakeholder Environments (RMSE) at ACM RecSys*, 2019.`,
  `K. Järvelin and J. Kekäläinen, "Cumulated gain-based evaluation of IR techniques," *ACM Transactions on Information Systems*, vol. 20, no. 4, 2002.`,
];
refs.forEach((t, i) => body.push(new Paragraph({ spacing: { after: 90, line: 270 }, indent: { left: 540, hanging: 540 }, alignment: AlignmentType.LEFT,
  children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 21 }), ...runs(t, { size: 21 })], tabStops: [{ type: "left", position: 540 }] })));

// =================== APPENDICES ===================
body.push(pb());
body.push(H1("Appendix A: Hyper-parameter Search Spaces and Selected Values"));
body.push(tcap("Table A1: Tuning grids (validation split) and selected values"));
body.push(table([2400, 4100, 2526], [
  ["Model", "Search space", "Selected"],
  ["User-KNN", "k ∈ {5, 10, 20, 50, 100, 200}; δ ∈ {0.05, 0.2, 0.5, 1, 2}", `k = ${B.user_k}, δ = ${B.damp}`],
  ["Item-KNN", "k ∈ {5, 10, 20, 50, 100, 200}; δ as above", `k = ${B.item_k}, δ = ${B.damp}`],
  ["SVD (residual)", "rank ∈ {5, 10, 20, 40, 80}", `rank = ${B.svd_k}`],
  ["ALS MF", "factors ∈ {16, 32, 64}; λ ∈ {0.15, 0.2, 0.3}; 10 iterations during tuning, 15 for final fit", `f = ${B.als_k}, λ = ${B.als_lam}`],
  ["Item-KNN (implicit)", "k ∈ {10, 20, 50, 100, 200}", `k = ${B.imp_k}`],
  ["PureSVD", "rank ∈ {10, 25, 50, 100, 200}", `rank = ${B.psvd_k}`],
  ["Hybrid weights", "12 weight triples (w₁, w₂, w₃) over PureSVD, Item-KNN impl., Content", `(${B.hybrid_w.join(", ")})`],
  ["Fixed", "Bias damping λ = 10; similarity shrinkage λ_s = 20 (explicit), 10 (implicit); relevance threshold ≥ 8; seed 42", "—"],
], { left: [0, 1, 2], size: 18 }));
body.push(H1("Appendix B: Code Structure and Reproduction Steps"));
body.push(P(`The submitted code archive contains:`));
body.push(bullet(`**recsys_lib.py** – data loading, k-core filtering, splitting, the eleven recommenders and the evaluation functions.`));
body.push(bullet(`**run_experiments.py** – the complete pipeline: exploratory plots, tuning on validation, final test evaluation, significance tests, segment analysis, qualitative example, figures and results.json.`));
body.push(bullet(`**build_report.js** – regenerates this report from results.json.`));
body.push(bullet(`**results/** – results.json and all figures (PNG).`));
body.push(P(`**To reproduce:** download ratings.dat and movies.dat from the MovieTweetings repository [21] into a folder named data, install the requirements (numpy, scipy, pandas, matplotlib), and run python run_experiments.py. Results are deterministic given the seed (42).`, { before: 100 }));

// =================== ASSEMBLE ===================
const doc = new Document({
  creator: "Rhea", title: "Comparative Analysis of Existing Recommender Systems",
  styles: {
    default: { document: { run: { font: FONT, size: 23 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, font: FONT, color: "1F3864" }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 25, bold: true, font: FONT, color: "2F5496" }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
  ] },
  sections: [
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } }, children: cover },
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1300, left: 1440 }, pageNumbers: { start: 1 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] }) },
      children: body },
  ],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync("Recommender_Systems_Project_Report.docx", b); console.log("written", b.length); });

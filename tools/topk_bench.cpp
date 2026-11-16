/*
 * topk_bench -- the shipped selection against a two-pass rewrite.
 *
 * The shipped selection (fl_topk in minivectordb/_fastlane.cpp) answers a
 * TopK query with four passes over the column:
 *
 *   1. build the squared distance of every row      (a 40 KB allocation)
 *   2. sample the column and pick a threshold        (a strided pass)
 *   3. collect every row below the threshold         (a 40 KB allocation)
 *      and nth_element over the survivors
 *   4. collect the rows that tie with the boundary
 *
 * The rewrite (fl_topk_v2) answers the same question with two passes:
 *
 *   1. one pass that keeps the `take` smallest values in a bounded max-heap
 *      and takes the boundary from the heap
 *   2. one pass that collects the band and computes the exact distances
 *
 * The two must agree exactly, not approximately: the band decides which rows
 * get their exact float64 distance computed, so a different band is a
 * different answer.  This bench checks the two answers against each other,
 * bit for bit, on random tables of several shapes, and then measures both.
 *
 * The benchmark is a development tool; it is not part of the shipped
 * library.  Build it with:
 *
 *   g++ -O3 -std=c++17 -ffp-contract=off -fno-fast-math \
 *       -I minivectordb tools/topk_bench.cpp -o tmp/topk_bench
 */
#include "_fastlane.cpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <string>

/* The two-pass rewrite, declared exactly like the shipped one so the two
 * can be called with the same arguments. */
extern "C" int64_t fl_topk_v2(const float *coarse, const float *norms,
                              const float *matrix, const float *query,
                              const int64_t *keys, const int64_t *rows_of,
                              const int64_t *ids_of,
                              int64_t id_count, int64_t dim, int64_t take,
                              int64_t offset, double tolerance,
                              int64_t *out_ids, double *out_dists)
{
    if (coarse == NULL || take <= 0 || id_count < 0) return -1;
    if (take > id_count) take = id_count;
    if (take <= 0) return 0;

    /* The boundary: the take-th smallest of |a|^2 - 2 a.q over the column.
     *
     * A bounded max-heap holds the `take` smallest values seen so far, so
     * the heap's root is the take-th smallest of the whole column once the
     * pass ends.  The comparison that decides whether a row enters the heap
     * is almost always false -- only about take*log(count/take) rows of ten
     * thousand enter it -- so the branch predicts well, which is exactly
     * what the partition this replaces could not do.
     *
     * The values are the same bits the shipped kernel produced: the product
     * is multiplied by -2 and the norm is added, in that order, in float32,
     * with the compiler forbidden from fusing the two into an FMA. */
    std::vector<float> heap;
    heap.reserve(static_cast<size_t>(take));
    for (int64_t i = 0; i < id_count; i++) {
        float value = coarse[i] * -2.0f;
        value = norms ? value + norms[i] : value;
        if (static_cast<int64_t>(heap.size()) < take) {
            heap.push_back(value);
            std::push_heap(heap.begin(), heap.end());
        } else if (value < heap.front()) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = value;
            std::push_heap(heap.begin(), heap.end());
        }
    }
    float boundary = heap.front();

    /* The band: every row that ties with the boundary.  The width of the
     * band is the same tolerance the shipped kernel used, and the exact
     * distance of every row in it is computed in the same order, so the
     * answer is the same answer. */
    float limit = static_cast<float>(static_cast<double>(boundary) + tolerance);
    std::vector<Candidate> band;
    band.reserve(static_cast<size_t>(take) * 2 + 8);
    for (int64_t i = 0; i < id_count; i++) {
        float value = coarse[i] * -2.0f;
        value = norms ? value + norms[i] : value;
        if (value > limit) continue;
        int64_t row = rows_of ? rows_of[i] : i;
        double distance = 0.0;
        if (matrix && query) {
            const float *values = matrix + static_cast<size_t>(row)
                                        * static_cast<size_t>(dim);
            double acc = 0.0;
            for (int64_t j = 0; j < dim; j++) {
                double diff = static_cast<double>(values[j])
                              - static_cast<double>(query[j]);
                acc += diff * diff;
            }
            distance = sqrt(acc);
        }
        Candidate candidate;
        candidate.distance = distance;
        candidate.key = keys ? keys[i] : row;
        candidate.row = ids_of ? ids_of[i] : row;
        band.push_back(candidate);
    }
    if (band.empty()) return 0;
    std::sort(band.begin(), band.end(), candidate_less);

    int64_t written = 0;
    for (size_t i = 0; i < band.size(); i++) {
        out_ids[written] = band[i].row;
        out_dists[written] = band[i].distance;
        written++;
    }
    return written;
}

/* ------------------------------------------------------------------ */

struct Table {
    int64_t rows = 0;
    int64_t dim = 0;
    std::vector<float> matrix;
    std::vector<float> norms;
    std::vector<float> coarse;
    std::vector<int64_t> keys;
};

static void build(Table &table, int64_t rows, int64_t dim,
                  unsigned seed, bool clustered)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    table.rows = rows;
    table.dim = dim;
    table.matrix.resize(static_cast<size_t>(rows * dim));
    table.norms.resize(static_cast<size_t>(rows));
    table.coarse.resize(static_cast<size_t>(rows));
    table.keys.resize(static_cast<size_t>(rows));
    for (int64_t r = 0; r < rows; r++) {
        float *row = table.matrix.data() + r * dim;
        float norm = 0.0f;
        for (int64_t j = 0; j < dim; j++) {
            /* A clustered table has rows that sit near each other, which is
             * the case where many rows tie with the boundary and the band
             * grows; the uniform table is the other extreme. */
            row[j] = clustered ? normal(rng) : uniform(rng);
            norm += row[j] * row[j];
        }
        table.norms[static_cast<size_t>(r)] = norm;
        table.keys[static_cast<size_t>(r)] = r + 1;
    }
    std::vector<float> query(static_cast<size_t>(dim));
    for (int64_t j = 0; j < dim; j++) {
        query[static_cast<size_t>(j)] = clustered ? normal(rng) : uniform(rng);
    }
    for (int64_t r = 0; r < rows; r++) {
        const float *row = table.matrix.data() + r * dim;
        float acc = 0.0f;
        for (int64_t j = 0; j < dim; j++) {
            acc += row[j] * query[static_cast<size_t>(j)];
        }
        table.coarse[static_cast<size_t>(r)] = acc;
    }
}

static bool same(const std::vector<int64_t> &ids_a,
                 const std::vector<double> &dists_a,
                 const std::vector<int64_t> &ids_b,
                 const std::vector<double> &dists_b)
{
    if (ids_a.size() != ids_b.size()) return false;
    for (size_t i = 0; i < ids_a.size(); i++) {
        if (ids_a[i] != ids_b[i]) return false;
        if (dists_a[i] != dists_b[i]) return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    const int64_t dim = 128;
    const int64_t take = 10;
    const double tolerance = 1e-3;
    const int rounds = argc > 1 ? atoi(argv[1]) : 200;

    /* ---- the answers must agree, bit for bit ----------------------- */
    struct Shape { int64_t rows; bool clustered; };
    const Shape shapes[] = {
        {10000, false}, {10000, true}, {1250, false}, {1250, true},
        {1, false}, {2, false}, {17, false}, {100, true}, {999, false},
        {20000, false},
    };
    int checked = 0;
    int mismatched = 0;
    for (const Shape &shape : shapes) {
        for (unsigned seed = 1; seed <= 4; seed++) {
            Table table;
            build(table, shape.rows, dim, seed, shape.clustered);
            std::vector<float> query(static_cast<size_t>(dim));
            std::mt19937 rng(seed * 7919u);
            std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
            for (int64_t j = 0; j < dim; j++) {
                query[static_cast<size_t>(j)] = uniform(rng);
            }
            std::vector<int64_t> ids_a(static_cast<size_t>(shape.rows));
            std::vector<double> dists_a(static_cast<size_t>(shape.rows));
            std::vector<int64_t> ids_b(static_cast<size_t>(shape.rows));
            std::vector<double> dists_b(static_cast<size_t>(shape.rows));
            int64_t written_a = fl_topk(
                table.coarse.data(), table.norms.data(), table.matrix.data(),
                query.data(), table.keys.data(), NULL, NULL,
                shape.rows, dim, take, 0, tolerance,
                ids_a.data(), dists_a.data());
            int64_t written_b = fl_topk_v2(
                table.coarse.data(), table.norms.data(), table.matrix.data(),
                query.data(), table.keys.data(), NULL, NULL,
                shape.rows, dim, take, 0, tolerance,
                ids_b.data(), dists_b.data());
            std::vector<int64_t> head_a(ids_a.begin(), ids_a.begin() + written_a);
            std::vector<double> head_d(dists_a.begin(), dists_a.begin() + written_a);
            std::vector<int64_t> head_b(ids_b.begin(), ids_b.begin() + written_b);
            std::vector<double> head_e(dists_b.begin(), dists_b.begin() + written_b);
            checked++;
            if (!same(head_a, head_d, head_b, head_e)) {
                mismatched++;
                printf("MISMATCH rows=%lld clustered=%d seed=%u "
                       "(shipped %lld rows, rewrite %lld rows)\n",
                       (long long)shape.rows, (int)shape.clustered, seed,
                       (long long)written_a, (long long)written_b);
            }
        }
    }
    printf("agreement: %d cases, %d mismatches\n", checked, mismatched);
    if (mismatched) return 1;

    /* ---- the speed ------------------------------------------------- */
    Table table;
    build(table, 10000, dim, 20261112, false);
    std::vector<float> query(static_cast<size_t>(dim));
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    for (int64_t j = 0; j < dim; j++) {
        query[static_cast<size_t>(j)] = uniform(rng);
    }
    std::vector<int64_t> ids(static_cast<size_t>(10000));
    std::vector<double> dists(static_cast<size_t>(10000));

    for (int which = 0; which < 2; which++) {
        /* One warm-up round, then the measured ones. */
        for (int i = 0; i < 8; i++) {
            if (which == 0) {
                fl_topk(table.coarse.data(), table.norms.data(),
                        table.matrix.data(), query.data(), table.keys.data(),
                        NULL, NULL, 10000, dim, take, 0, tolerance,
                        ids.data(), dists.data());
            } else {
                fl_topk_v2(table.coarse.data(), table.norms.data(),
                           table.matrix.data(), query.data(), table.keys.data(),
                           NULL, NULL, 10000, dim, take, 0, tolerance,
                           ids.data(), dists.data());
            }
        }
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < rounds; i++) {
            if (which == 0) {
                fl_topk(table.coarse.data(), table.norms.data(),
                        table.matrix.data(), query.data(), table.keys.data(),
                        NULL, NULL, 10000, dim, take, 0, tolerance,
                        ids.data(), dists.data());
            } else {
                fl_topk_v2(table.coarse.data(), table.norms.data(),
                           table.matrix.data(), query.data(), table.keys.data(),
                           NULL, NULL, 10000, dim, take, 0, tolerance,
                           ids.data(), dists.data());
            }
        }
        auto stop = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(stop - start).count()
                    / rounds;
        printf("%-12s %8.2f us/select\n", which == 0 ? "shipped" : "rewrite", us);
    }
    return 0;
}

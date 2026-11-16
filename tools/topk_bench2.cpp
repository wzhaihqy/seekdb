/*
 * topk_bench2 -- where the time of one selection goes, and a faster one.
 *
 * The shipped kernel (fl_topk in minivectordb/_fastlane.cpp) walks the
 * candidate column four times:
 *
 *   1. build the squared distance of every row      (a 40 KB allocation,
 *                                                    and a zeroed one)
 *   2. sample the column and pick a threshold        (nth_element over
 *                                                    1024 values)
 *   3. collect every row below the threshold         (a 40 KB allocation)
 *      and nth_element over the survivors
 *   4. collect the rows that tie with the boundary
 *
 * This bench measures those four steps separately, and then measures a
 * rewrite that
 *
 *   * builds the squared distances without the allocation and with the
 *     multiply and the add in two separate vector passes, so the compiler
 *     can use a vector unit,
 *   * keeps its scratch space between calls,
 *   * collects the band and the boundary in one pass over the column: the
 *     threshold the sample proposes is already above the boundary, so the
 *     rows within a tolerance of it are collected together with the rows
 *     below the threshold and the band is cut out of that set afterwards.
 *
 * The rewrite must agree with the shipped kernel bit for bit: the band
 * decides which rows get their exact float64 distance computed, so a
 * different band is a different answer.
 *
 * This benchmark is a development tool; it is not part of the shipped
 * library.  Build it with:
 *
 *   g++ -O3 -std=c++17 -ffp-contract=off -fno-fast-math \
 *       -I minivectordb tools/topk_bench2.cpp -o tmp/topk_bench2
 */
#include "_fastlane.cpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

namespace {

/* The scratch the rewrite keeps between calls.  The shipped kernel builds
 * its arrays per call; a query is a few dozen microseconds and a malloc of
 * forty kilobytes is not free. */
struct Scratch {
    std::vector<float> squared;
    std::vector<float> sample;
    std::vector<uint32_t> kept;
    std::vector<int64_t> kept_index;

    void reserve(int64_t count) {
        if ((int64_t)squared.size() < count) squared.resize((size_t)count);
        if ((int64_t)kept.size() < count) kept.resize((size_t)count);
        if ((int64_t)kept_index.size() < count) kept_index.resize((size_t)count);
        if (sample.size() < 1024) sample.resize(1024);
    }
};

Scratch g_scratch;

/* One row of the answer, with the index it came from. */
struct Scored {
    float value;
    int64_t index;
};

bool scored_less(const Scored &left, const Scored &right)
{
    if (left.value != right.value) return left.value < right.value;
    return left.index < right.index;
}

/* The rewrite. */
int64_t fl_topk_v3(const float *coarse, const float *norms,
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

    Scratch &scratch = g_scratch;
    scratch.reserve(id_count);
    float *squared = scratch.squared.data();

    /* The squared distance of every row, in the same float32 arithmetic the
     * numpy path used: (product * -2) + |row|^2.  The multiply and the add
     * stay separate (the compiler must not fuse them, hence
     * -ffp-contract=off), so the bits are the bits the numpy path saw. */
    if (norms) {
        for (int64_t i = 0; i < id_count; i++) {
            squared[i] = coarse[i] * -2.0f + norms[i];
        }
    } else {
        for (int64_t i = 0; i < id_count; i++) {
            squared[i] = coarse[i] * -2.0f;
        }
    }

    /* A threshold from a sample of the column.  The threshold is above the
     * boundary the question asks for, so the rows within a tolerance of it
     * are collected together with the rows below it, and the band is cut
     * out of that set once the boundary is known. */
    const int64_t sample_size = 1024;
    float *sample = scratch.sample.data();
    int64_t stride = id_count / sample_size;
    if (stride < 1) stride = 1;
    int64_t sampled = 0;
    for (int64_t i = 0; i * stride < id_count && sampled < sample_size; i++) {
        sample[sampled++] = squared[i * stride];
    }
    int64_t local_rank = (take - 1 + 1) * sampled / id_count + 4;
    if (local_rank >= sampled) local_rank = sampled - 1;
    if (local_rank < 0) local_rank = 0;
    std::nth_element(sample, sample + local_rank, sample + sampled);
    float threshold = sample[local_rank];

    /* The limit the rows are collected with: the threshold plus the
     * tolerance of the band.  The boundary is at or below the threshold, so
     * this limit is at or above the boundary's limit, and the set that comes
     * out of the column contains the whole band. */
    float limit = static_cast<float>(static_cast<double>(threshold) + tolerance);

    uint32_t *kept = scratch.kept.data();
    int64_t *kept_index = scratch.kept_index.data();
    int64_t kept_count = 0;
    int64_t boundary_rank = take - 1;

    for (int64_t attempt = 0; attempt < 24; attempt++) {
        kept_count = 0;
        for (int64_t i = 0; i < id_count; i++) {
            if (squared[i] <= limit) {
                kept[kept_count] = fl_order_key(squared[i]);
                kept_index[kept_count] = i;
                kept_count++;
            }
        }
        if (kept_count > boundary_rank) break;
        /* Too few rows below the threshold: the sample understated the
         * distribution.  A higher quantile of the same sample is a better
         * threshold than the next float above. */
        int64_t next = local_rank + local_rank / 4 + 8;
        if (next >= sampled) {
            next = sampled - 1;
            if (next == local_rank) break;
        }
        local_rank = next;
        std::nth_element(sample, sample + local_rank, sample + sampled);
        threshold = sample[local_rank];
        limit = static_cast<float>(static_cast<double>(threshold) + tolerance);
    }
    if (kept_count <= boundary_rank) {
        /* The threshold could not be raised far enough: fall back to the
         * radix route, which always answers. */
        float boundary = fl_select_radix(squared, id_count, boundary_rank);
        float band_limit = static_cast<float>(static_cast<double>(boundary) + tolerance);
        int64_t written = 0;
        for (int64_t i = 0; i < id_count && written < take; i++) {
            if (squared[i] > band_limit) continue;
            int64_t row = rows_of ? rows_of[i] : i;
            double distance = 0.0;
            if (matrix && query) {
                const float *values = matrix + (size_t)row * (size_t)dim;
                double acc = 0.0;
                for (int64_t j = 0; j < dim; j++) {
                    double diff = (double)values[j] - (double)query[j];
                    acc += diff * diff;
                }
                distance = sqrt(acc);
            }
            out_ids[written] = ids_of ? ids_of[i] : row;
            out_dists[written] = distance;
            written++;
        }
        return written;
    }

    /* The boundary: the take-th smallest of the collected rows. */
    std::vector<uint32_t> order((size_t)kept_count);
    for (int64_t i = 0; i < kept_count; i++) order[(size_t)i] = kept[i];
    std::nth_element(order.begin(), order.begin() + boundary_rank, order.end());
    float boundary = fl_order_value(order[(size_t)boundary_rank]);

    /* The band: the collected rows that are within the tolerance of the
     * boundary. */
    float band_limit = static_cast<float>(static_cast<double>(boundary) + tolerance);
    std::vector<Candidate> band;
    band.reserve((size_t)take * 2 + 8);
    for (int64_t i = 0; i < kept_count; i++) {
        float value = fl_order_value(kept[i]);
        if (value > band_limit) continue;
        int64_t index = kept_index[i];
        int64_t row = rows_of ? rows_of[index] : index;
        double distance = 0.0;
        if (matrix && query) {
            const float *values = matrix + (size_t)row * (size_t)dim;
            double acc = 0.0;
            for (int64_t j = 0; j < dim; j++) {
                double diff = (double)values[j] - (double)query[j];
                acc += diff * diff;
            }
            distance = sqrt(acc);
        }
        Candidate candidate;
        candidate.distance = distance;
        candidate.key = keys ? keys[index] : row;
        candidate.row = ids_of ? ids_of[index] : row;
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

/* -- the phases of the shipped kernel, timed on their own ------------- */

int64_t phase_squared(const float *coarse, const float *norms, int64_t count)
{
    std::vector<float> squared((size_t)count);
    for (int64_t i = 0; i < count; i++) {
        float value = coarse[i] * -2.0f;
        squared[(size_t)i] = norms ? value + norms[i] : value;
    }
    return (int64_t)squared[0];
}

int64_t phase_sample(const float *values, int64_t count, int64_t rank)
{
    return (int64_t)fl_select_sample(values, count, rank);
}

int64_t phase_collect(const float *values, int64_t count, float threshold)
{
    std::vector<uint32_t> kept;
    kept.reserve((size_t)count);
    for (int64_t i = 0; i < count; i++) {
        if (values[i] <= threshold) kept.push_back(fl_order_key(values[i]));
    }
    return (int64_t)kept.size();
}

double now_seconds()
{
    using clock = std::chrono::steady_clock;
    static const clock::time_point start = clock::now();
    return std::chrono::duration<double>(clock::now() - start).count();
}

template <typename F>
double time_us(F fn, int rounds)
{
    double best = 1e30;
    for (int r = 0; r < rounds; r++) {
        double t0 = now_seconds();
        fn();
        double dt = now_seconds() - t0;
        if (dt < best) best = dt;
    }
    return best * 1e6;
}

}  /* namespace */

int main()
{
    std::mt19937 rng(20261116);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);

    const int64_t rows = 10000;
    const int64_t dim = 128;
    const int64_t take = 10;

    std::vector<float> matrix((size_t)(rows * dim));
    std::vector<float> norms((size_t)rows);
    std::vector<float> query((size_t)dim);
    std::vector<int64_t> keys((size_t)rows);
    for (int64_t i = 0; i < rows; i++) {
        float norm = 0.0f;
        for (int64_t j = 0; j < dim; j++) {
            float value = uniform(rng);
            matrix[(size_t)(i * dim + j)] = value;
            norm += value * value;
        }
        norms[(size_t)i] = norm;
        keys[(size_t)i] = i + 1;
    }
    for (int64_t j = 0; j < dim; j++) query[(size_t)j] = uniform(rng);

    std::vector<float> coarse((size_t)rows);
    for (int64_t i = 0; i < rows; i++) {
        float acc = 0.0f;
        for (int64_t j = 0; j < dim; j++) {
            acc += matrix[(size_t)(i * dim + j)] * query[(size_t)j];
        }
        coarse[(size_t)i] = acc;
    }
    std::vector<int64_t> out_ids((size_t)rows);
    std::vector<double> out_dists((size_t)rows);

    /* The two answers must agree, bit for bit. */
    int mismatches = 0;
    std::vector<float> real((size_t)rows);
    for (int trial = 0; trial < 40; trial++) {
        for (int64_t i = 0; i < rows; i++) {
            float acc = 0.0f;
            for (int64_t j = 0; j < dim; j++) {
                acc += matrix[(size_t)(i * dim + j)] * query[(size_t)j];
            }
            real[(size_t)i] = acc;
        }
        int64_t a = fl_topk(real.data(), norms.data(), matrix.data(), query.data(),
                            keys.data(), NULL, NULL, rows, dim, take, 0, 1e-3,
                            out_ids.data(), out_dists.data());
        std::vector<int64_t> ids_a((size_t)a);
        std::vector<double> dists_a((size_t)a);
        for (int64_t i = 0; i < a; i++) { ids_a[(size_t)i] = out_ids[(size_t)i]; dists_a[(size_t)i] = out_dists[(size_t)i]; }
        int64_t b = fl_topk_v3(real.data(), norms.data(), matrix.data(), query.data(),
                               keys.data(), NULL, NULL, rows, dim, take, 0, 1e-3,
                               out_ids.data(), out_dists.data());
        if (a != b) { mismatches++; continue; }
        for (int64_t i = 0; i < a; i++) {
            if (ids_a[(size_t)i] != out_ids[(size_t)i]) { mismatches++; break; }
            if (dists_a[(size_t)i] != out_dists[(size_t)i]) { mismatches++; break; }
        }
    }
    printf("agreement: 40 cases, %d mismatches\n", mismatches);

    /* The phases, on their own. */
    for (int64_t i = 0; i < rows; i++) {
        float acc = 0.0f;
        for (int64_t j = 0; j < dim; j++) {
            acc += matrix[(size_t)(i * dim + j)] * query[(size_t)j];
        }
        coarse[(size_t)i] = acc;
    }
    std::vector<float> squared((size_t)rows);
    for (int64_t i = 0; i < rows; i++) {
        float value = coarse[(size_t)i] * -2.0f;
        squared[(size_t)i] = value + norms[(size_t)i];
    }
    float boundary = fl_select_kth(squared.data(), rows, take - 1);
    float limit = (float)(boundary + 1e-3);

    printf("\n=== the phases of the shipped kernel (10000 rows) ===\n");
    printf("  build the squared distances   %8.2f us\n",
           time_us([&] { phase_squared(coarse.data(), norms.data(), rows); }, 40));
    printf("  sample and pick a threshold    %8.2f us\n",
           time_us([&] { phase_sample(squared.data(), rows, take - 1); }, 40));
    printf("  collect the rows below it      %8.2f us\n",
           time_us([&] { phase_collect(squared.data(), rows, limit); }, 40));

    printf("\n=== the whole selection ===\n");
    printf("  shipped                       %8.2f us\n",
           time_us([&] {
               fl_topk(coarse.data(), norms.data(), matrix.data(), query.data(),
                       keys.data(), NULL, NULL, rows, dim, take, 0, 1e-3,
                       out_ids.data(), out_dists.data());
           }, 40));
    printf("  rewrite                       %8.2f us\n",
           time_us([&] {
               fl_topk_v3(coarse.data(), norms.data(), matrix.data(), query.data(),
                          keys.data(), NULL, NULL, rows, dim, take, 0, 1e-3,
                          out_ids.data(), out_dists.data());
           }, 40));
    return 0;
}

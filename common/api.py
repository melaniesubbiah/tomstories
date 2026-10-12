"""Anthropic/OpenAI helpers: rate-limit backoff and the data-generation driver."""

import asyncio
import os
import random

from tqdm import tqdm

from common.data import save_json
from common.prompts import parse_scenario


def is_rate_limit(exc):
    name, msg = type(exc).__name__.lower(), str(exc).lower()
    return ("ratelimit" in name or "resourceexhausted" in name
            or "rate limit" in msg or "429" in msg)


async def with_backoff(coro_fn, max_retries=6, base_delay=1.0, max_delay=60.0):
    """Await coro_fn(), retrying rate-limit errors with jittered exponential backoff."""
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except Exception as e:
            if not is_rate_limit(e) or attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay + random.uniform(0, delay * 0.2))
            delay = min(delay * 2, max_delay)


async def map_async(items, worker, concurrency, desc):
    """Run worker(item) over items with bounded concurrency; results stay in input order."""
    semaphore = asyncio.Semaphore(concurrency)
    pbar      = tqdm(total=len(items), desc=desc)

    async def run(i, item):
        async with semaphore:
            result = await worker(item)
        pbar.update(1)
        return i, result

    pairs = await asyncio.gather(*[run(i, item) for i, item in enumerate(items)])
    pbar.close()
    out = [None] * len(items)
    for i, result in pairs:
        out[i] = result
    return out


def anthropic_client():
    from anthropic import AsyncAnthropic
    return AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


async def call_claude(client, model, system, prompt, max_tokens=1024, temperature=0.7):
    async def _call():
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.content[0].text

    return await with_backoff(_call)


async def generate_vignettes(pending, build_prompt, build_record, output_path, results,
                             model, concurrency, desc, temperature=0.7):
    """
    Generate one vignette per pending item, saving after each completion so runs resume.

    pending      — [(sample_id, spec), …] items still to generate
    build_prompt — spec -> (system prompt, user prompt)
    build_record — (spec, scenario, question) -> the dict to store for this sample
    """
    client    = anthropic_client()
    semaphore = asyncio.Semaphore(concurrency)
    lock      = asyncio.Lock()
    pbar      = tqdm(total=len(pending), desc=desc)

    async def process(sample_id, spec):
        system, prompt = build_prompt(spec)
        async with semaphore:
            try:
                output = await call_claude(client, model, system, prompt,
                                           temperature=temperature)
            except Exception as e:
                tqdm.write(f"\nError on {sample_id}: {e}")
                return
        scenario, question = parse_scenario(output)
        async with lock:
            results[sample_id] = build_record(spec, scenario, question)
            save_json(output_path, results, indent=2)
            pbar.update(1)

    await asyncio.gather(*[process(sid, spec) for sid, spec in pending])
    pbar.close()
    return results

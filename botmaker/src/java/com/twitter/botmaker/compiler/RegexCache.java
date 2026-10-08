package com.twitter.botmaker.compiler;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;

import com.twitter.botmaker.Context;
import com.twitter.botmaker.compiler.tuples.Pair;
import com.twitter.util.Try;
import com.twitter.util.Try$;

public interface RegexCache {

  boolean matches(Context<?> context, String regex, String str);

  List<String> extract(Context<?> context, String regex, String str);

  List<String> split(Context<?> context, String regex, String str);

  String replace(Context<?> context, String str, String target, String replacement);

  static String either(String... args) {
    return "(?:" + String.join("|", args) + ")";
  }


  static CharSequence interruptible(Context<?> context, CharSequence str, int timeoutMilis) {

    Timeout timeout = new Timeout() {
      @Override
      public String toString() {
        return String.format(
            "creationTimestamp=%d, timeoutMillis=%d",
            creationTimestamp, timeoutMilis);
      }

      private final long creationTimestamp = context.nowMillis();

      @Override
      public boolean isDone() {
        return context.nowMillis() > creationTimestamp + timeoutMilis;
      }
    };

    return new InterruptibleCharSequence(str, timeout);
  }

  static Matcher getInterruptibleMatcher(
      Context<?> context,
      Pattern pattern,
      CharSequence input
  ) {
    CharSequence interruptibleText =
        RegexCache.interruptible(context, input, RegexCache.DEFAULT_TIMEOUT_MS);
    return pattern.matcher(interruptibleText);
  }

  static RegexCache of(
      int timeoutMilis,
      LoadingCache<Pair<String, Integer>, Try<Pattern>> patterns
  ) {
    return new RegexCache() {
      @Override
      public boolean matches(Context<?> context, String regex, String str) {
        Pattern pattern = patterns.getUnchecked(Pair.of(regex, Pattern.DOTALL)).get();
        return pattern.matcher(interruptible(context, str, timeoutMilis)).matches();
      }

      @Override
      public List<String> extract(Context<?> context, String regex, String str) {
        Pattern pattern = patterns.getUnchecked(Pair.of(regex, 0)).get();
        Matcher m = pattern.matcher(interruptible(context, str, timeoutMilis));

        ImmutableList.Builder<String> builder = ImmutableList.builder();
        if (!m.find()) {
          return builder.build();
        }

        int groupCount = m.groupCount();
        for (int i = 1; i <= groupCount; i++) {
          builder.add(m.group(i));
        }

        return builder.build();
      }

      @Override
      public List<String> split(Context<?> context, String regex, String str) {

        int limit = 0;
        char ch = 0;
        if (((regex.length() == 1
            && ".$|()[{^?*+\\".indexOf(ch = regex.charAt(0)) == -1)
            || (regex.length() == 2
            && regex.charAt(0) == '\\'
            && (((ch = regex.charAt(1)) - '0') | ('9' - ch)) < 0
            && ((ch - 'a') | ('z' - ch)) < 0
            && ((ch - 'A') | ('Z' - ch)) < 0))
            && (ch < Character.MIN_HIGH_SURROGATE || ch > Character.MAX_LOW_SURROGATE)) {
          int off = 0;
          int next = 0;
          boolean limited = limit > 0;
          ArrayList<String> list = new ArrayList<>();
          while ((next = str.indexOf(ch, off)) != -1) {
            if (!limited || list.size() < limit - 1) {
              list.add(str.substring(off, next));
              off = next + 1;
            } else {
              list.add(str.substring(off, str.length()));
              off = str.length();
              break;
            }
          }
          if (off == 0) {
            return ImmutableList.of(str);
          }

          if (!limited || list.size() < limit) {
            list.add(str.substring(off, str.length()));
          }

          int resultSize = list.size();
          if (limit == 0) {
            while (resultSize > 0 && list.get(resultSize - 1).length() == 0) {
              resultSize--;
            }
          }
          return ImmutableList.copyOf(list.subList(0, resultSize));
        }

        Pattern pattern = patterns.getUnchecked(Pair.of(regex, 0)).get();
        return ImmutableList.copyOf(pattern.split(interruptible(context, str, timeoutMilis)));
      }

      @Override
      public String replace(Context<?> context, String str, String target, String replacement) {
        Pattern pattern = patterns.getUnchecked(Pair.of(target, 0)).get();
        return pattern.matcher(interruptible(context, str, timeoutMilis))
            .replaceAll(replacement);
      }
    };
  }

  int DEFAULT_TIMEOUT_MS = 1000;

  RegexCache DEFAULT = of(
      DEFAULT_TIMEOUT_MS,
      CacheBuilder.newBuilder()
          .maximumSize(1000)
          .expireAfterAccess(30, TimeUnit.MINUTES)
          .build(CacheLoader.from((Pair<String, Integer> key) ->
              Try$.MODULE$.apply(() -> Pattern.compile(key.getFirst(), key.getSecond()))
          ))
  );
}

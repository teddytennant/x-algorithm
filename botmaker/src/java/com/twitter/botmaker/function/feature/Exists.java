package com.twitter.botmaker.function.feature;

import com.google.common.collect.ImmutableList;
import com.google.common.hash.Hasher;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.Extractor;
import com.twitter.botmaker.Extractor.Functor;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.runtime.Runtime;
import com.twitter.util.Future;

public abstract class Exists extends ASTNode<Runtime> {
  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(),
      Type.BOOLEAN
  );

  private final String inputFeatureName;

  private Exists(String exprText, String inputFeatureName) throws SemanticCheckFailure {
    super(exprText, ImmutableList.of());
    this.inputFeatureName = inputFeatureName;
  }

  @Override
  public Signature getSignature() {
    return SIGNATURE;
  }

  @Override
  protected CacheLevel getCacheLevel() {
    return CacheLevel.Event;
  }

  @Override
  public Extractor<Runtime> toExtractor() {
    return new Functor<Runtime>(this, ImmutableList.of()) {

      @Override
      public Future<Object> run(Context<Runtime> context) {
        return Future.value(apply(context));
      }

      @Override
      public Object apply(Context<Runtime> context) {
        return context.isInputFeatureDefined(inputFeatureName);
      }
    };
  }

  public static Exists of(String exprText, String inputFeatureName) throws SemanticCheckFailure {
    String internedInputFeatureName = intern(inputFeatureName);
    return new Exists(exprText, internedInputFeatureName) {
      @Override
      protected void computeFingerprint(Hasher hasher) {
        super.computeFingerprint(hasher);
        hasher.putUnencodedChars(internedInputFeatureName);
      }
    };
  }
}

package com.twitter.botmaker.function.misc;

import java.util.List;
import java.util.Map;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.Extractor;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.tuples.OutputKey;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.runtime.Runtime;
import com.twitter.util.Function;
import com.twitter.util.Future;

public class TestRun extends ASTNode<Runtime> {

  public static final Object NAMESPACE = new Object() {
    @Override
    public String toString() {
      return "TestRun";
    }
  };

  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(Type.OBJECT),
      Type.mapOf(
          Type.structTupleOf(
              OutputKey.FIELD_NAMES,
              ImmutableList.of(
                  Type.STRING,
                  Type.STRING,
                  Type.STRING
              )
          ),
          Type.OBJECT
      )
  );

  public TestRun(String exprText, ImmutableList<ASTNode> children)
      throws SemanticCheckFailure {
    super(exprText, children);
  }

  @Override
  public Signature getSignature() {
    return SIGNATURE;
  }

  @Override
  public Extractor<Runtime> toExtractor() {

    List<ASTNode> childNodes = getChildren();
    ImmutableList<Extractor> childExtractors = buildExtractorsOfChildren();

    return new Extractor<Runtime>(this, childExtractors) {

      @Override
      public Future<Object> run(Context<Runtime> context) {
        Map<OutputKey, Object> map = Maps.newConcurrentMap();
        Context<Runtime> ctx = new Context.Proxy<Runtime>(context) {

          @Override
          public void addOutputFeature(Object namespace, String name, Type type, Object value) {
            map.put(new OutputKey(namespace.toString(), type.toString(), name), value);
            super.addOutputFeature(namespace, name, type, value);
          }
        };

        return ctx.run(childExtractors.get(0)).map(Function.func(obj -> {
          OutputKey outputKey = new OutputKey(
              NAMESPACE.toString(),
              childNodes.get(0).getReturnType().toString(),
              "Result");
          map.put(outputKey, obj);
          return map;
        }));
      }
    };
  }

  @Override
  protected CacheLevel getCacheLevel() {
    return CacheLevel.Event;
  }
}

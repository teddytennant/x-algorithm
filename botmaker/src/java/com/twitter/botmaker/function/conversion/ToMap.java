package com.twitter.botmaker.function.conversion;

import java.util.Collection;
import java.util.Map;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.compiler.ActionLevel;
import com.twitter.botmaker.compiler.BotMakerFunction;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.tuples.Pair;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.function.FunctionNode1;
import com.twitter.botmaker.function.collection.ListOperator;
import com.twitter.botmaker.runtime.Runtime;

@BotMakerFunction(
    name = {"ToMap", "ToDict"},
    argTypes = {"Collection<Pair<A, B>>"},
    arguments = {"key / value"},
    returnType = "Map<A, B>",
    description = "Makes a list of EVEN number of objects into a map , 0 param is allowed",
    actionLevel = ActionLevel.NO_ACTION,
    examples = {
        "ToMap(123, \"asdf\", 456, \"jkl;\")"
    }
)
public class ToMap extends FunctionNode1<Runtime, Collection<Pair>> {

  private static final Type TPA = Type.newGenericType();
  private static final Type TPB = Type.newGenericType();
  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(Type.collectionOf(Type.pairOf(TPA, TPB))),
      Type.mapOf(TPA, TPB)
  );

  public ToMap(String exprText, ImmutableList<ASTNode> children)
      throws SemanticCheckFailure {
    super(exprText, convertParamsToPairs(children));
  }

  @Override
  public Signature getSignature() {
    return SIGNATURE;
  }

  @Override
  protected CacheLevel getCacheLevel() {
    return CacheLevel.Global;
  }

  private static ImmutableList<ASTNode> convertParamsToPairs(
      ImmutableList<ASTNode> children) throws SemanticCheckFailure {
    if (children.size() == 1) {
      ASTNode child = children.get(0);
      if (!Type.isDivergentTo(
          child.getReturnType(),
          Type.collectionOf(Type.pairOf(Type.OBJECT, Type.OBJECT)))) {
        return children;
      }
    }
    return ImmutableList.<ASTNode>of(
        new ListOperator("List", ToPair.convertNodesToPairs(children, 0)));
  }

  @Override
  @SuppressWarnings("unchecked")
  public Object apply(Context<Runtime> context, Collection<Pair> collection) {
    Map<Object, Object> hashMap = Maps.newHashMap();
    for (Pair<?, ?> pair : collection) {
      Object key = pair.getFirst();
      Object value = pair.getSecond();
      if (key != null && value != null) {
        hashMap.put(key, value);
      }
    }
    return ImmutableMap.copyOf(hashMap);
  }
}
